"""CodeAgent-specific preprocessing: rewrite bare tool calls to assignment + preview."""

import ast

_INVALID_VIEW_MESSAGE = "view() is a display tool, not a value. Use read() for file contents as text."
_DIRECT_READ_WARNING = "Direct file reads bypass code_agent context tools; prefer read() for file contents or view() for inspection."


def preprocess_code_agent(
    code: str,
    *,
    preview_targets: frozenset[str],
    preview_counter: int = 0,
    preview_origins: dict[str, str] | None = None,
) -> tuple[str, int]:
    """Rewrite bare tool calls into assignment + preview and guard value-style view().

    Returns the processed code and the updated preview counter.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code, preview_counter

    lines = code.split("\n")
    body = tree.body
    counter = preview_counter

    def _source(node):
        return ast.get_source_segment(code, node)

    def _is_named_call(node, *names):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        )

    def _has_literal_bg_true(call):
        return any(
            kw.arg == "bg"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in call.keywords
        )

    def _preview_uri_read_arg(call):
        if not _is_named_call(call, "read"):
            return None
        if call.keywords or len(call.args) != 1:
            return None
        arg = call.args[0]
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("session://preview/")
        ):
            return repr(arg.value)
        return None

    def _preview_uri_view_origin(call):
        if not _is_named_call(call, "view"):
            return None
        if call.keywords or len(call.args) != 1:
            return None
        arg = call.args[0]
        if (
            not isinstance(arg, ast.Constant)
            or not isinstance(arg.value, str)
            or not arg.value.startswith("session://preview/")
        ):
            return None
        origin = (preview_origins or {}).get(arg.value[len("session://preview/"):])
        return repr(origin) if origin is not None else None

    def _literal_path_expr(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return repr(node.value)
        if isinstance(node, ast.JoinedStr):
            return _source(node)
        return None

    def _is_path_constructor_call(node):
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name) and func.id == "Path":
            return True
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "Path"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pathlib"
        )

    def _path_constructor_arg(node):
        if not _is_path_constructor_call(node):
            return None
        if len(node.args) != 1 or node.keywords:
            return None
        return _literal_path_expr(node.args[0])

    def _is_any_path_constructor_call(node):
        return _is_path_constructor_call(node) and len(node.args) == 1 and not node.keywords

    def _direct_read_path(node):
        if not isinstance(node, ast.Call) or node.keywords:
            return None
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None

        if func.attr == "read_text" and not node.args:
            return _path_constructor_arg(func.value)

        if func.attr != "read" or node.args:
            return None

        value = func.value
        if _is_named_call(value, "open"):
            if len(value.args) == 1 and not value.keywords:
                return _literal_path_expr(value.args[0])
            return None

        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "open":
            if value.args or value.keywords:
                return None
            return _path_constructor_arg(value.func.value)

        return None

    def _is_direct_file_read(node):
        if isinstance(node, ast.Call):
            if _direct_read_path(node) is not None:
                return True
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"read_text", "read_bytes", "open"}
                and _is_any_path_constructor_call(node.func.value)
            ):
                return True
            if _is_named_call(node, "open"):
                return True
        return False

    def _contains_direct_file_read(node):
        return any(_is_direct_file_read(child) for child in ast.walk(node))

    def _warning_line(indent):
        return f"{indent}print({_DIRECT_READ_WARNING!r})"

    def _next_bash_var():
        nonlocal counter
        counter += 1
        return f"_bash{counter}"

    all_rewrites = []

    # Pattern 0: reject value-style uses of view(...)
    for node in body:
        start = node.lineno - 1
        end = node.end_lineno - 1
        orig = lines[start : end + 1]
        indent = orig[0][: len(orig[0]) - len(orig[0].lstrip())]

        if isinstance(node, ast.Assign) and _is_named_call(node.value, "view"):
            all_rewrites.append(
                (start, end, [f"{indent}raise ValueError({_INVALID_VIEW_MESSAGE!r})"])
            )
            continue

        if (
            isinstance(node, ast.AnnAssign)
            and node.value
            and _is_named_call(node.value, "view")
        ):
            all_rewrites.append(
                (start, end, [f"{indent}raise ValueError({_INVALID_VIEW_MESSAGE!r})"])
            )
            continue

        if (
            isinstance(node, ast.Expr)
            and _is_named_call(node.value, "preview", "print")
            and len(node.value.args) == 1
            and not node.value.keywords
        ):
            inner = node.value.args[0]
            if _is_named_call(inner, "view") or (
                isinstance(inner, ast.NamedExpr) and _is_named_call(inner.value, "view")
            ):
                all_rewrites.append(
                    (start, end, [f"{indent}raise ValueError({_INVALID_VIEW_MESSAGE!r})"])
                )
                continue

    # Pattern 1: direct file reads with clear read()/view() equivalents.
    for node in body:
        start = node.lineno - 1
        end = node.end_lineno - 1
        orig = lines[start : end + 1]
        indent = orig[0][: len(orig[0]) - len(orig[0].lstrip())]

        if isinstance(node, ast.Expr):
            origin_arg = _preview_uri_view_origin(node.value)
            if origin_arg is not None:
                all_rewrites.append((start, end, [f"{indent}view({origin_arg})"]))
                continue

            path_arg = _direct_read_path(node.value)
            if path_arg is not None:
                all_rewrites.append((start, end, [f"{indent}view({path_arg})"]))
                continue

            if _is_named_call(node.value, "print") and len(node.value.args) == 1 and not node.value.keywords:
                inner = node.value.args[0]
                path_arg = _direct_read_path(inner)
                if path_arg is not None:
                    all_rewrites.append((start, end, [f"{indent}view({path_arg})"]))
                    continue
                if isinstance(inner, ast.Subscript) and _direct_read_path(inner.value) is not None:
                    all_rewrites.append((start, start - 1, [_warning_line(indent)]))
                    continue

            if _contains_direct_file_read(node):
                all_rewrites.append((start, start - 1, [_warning_line(indent)]))
                continue

        value_node = None
        if isinstance(node, ast.Assign):
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and node.value:
            value_node = node.value

        if value_node is not None:
            path_arg = _direct_read_path(value_node)
            if path_arg is not None:
                source = _source(value_node)
                if source:
                    new_lines = list(orig)
                    rel_start = value_node.lineno - node.lineno
                    rel_end = value_node.end_lineno - node.lineno
                    prefix = new_lines[rel_start][:value_node.col_offset]
                    suffix = new_lines[rel_end][value_node.end_col_offset:]
                    replacement = f"read({path_arg})"
                    new_lines[rel_start : rel_end + 1] = [prefix + replacement + suffix]
                    all_rewrites.append((start, end, new_lines))
                    continue
            if _contains_direct_file_read(value_node):
                all_rewrites.append((start, start - 1, [_warning_line(indent)]))
                continue

    # Pattern 2: bare target call or print(target(...))
    for node in body:
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func

        start = node.lineno - 1
        end = node.end_lineno - 1
        orig = lines[start : end + 1]
        indent = orig[0][: len(orig[0]) - len(orig[0].lstrip())]

        if isinstance(func, ast.Name) and func.id in preview_targets:
            call_source = "\n".join(orig).strip()
            if func.id == "read":
                all_rewrites.append(
                    (start, end, [f"{indent}view{call_source[len('read'):]}"])
                )
            elif func.id == "bash":
                var_name = _next_bash_var()
                if _has_literal_bg_true(call):
                    all_rewrites.append(
                        (start, end, [f"{indent}{var_name} = {call_source}"])
                    )
                else:
                    all_rewrites.append(
                        (start, end, [f"{indent}preview({var_name} := {call_source})"])
                    )
            else:
                all_rewrites.append(
                    (start, end, [f"{indent}preview({call_source})"])
                )
        elif (
            isinstance(func, ast.Name)
            and func.id == "preview"
            and len(call.args) == 1
            and not call.keywords
            and _is_named_call(call.args[0], "bash")
        ):
            inner = call.args[0]
            inner_source = _source(inner)
            if inner_source:
                var_name = _next_bash_var()
                if _has_literal_bg_true(inner):
                    all_rewrites.append(
                        (start, end, [f"{indent}{var_name} = {inner_source}"])
                    )
                else:
                    all_rewrites.append(
                        (start, end, [f"{indent}preview({var_name} := {inner_source})"])
                    )
        elif (
            isinstance(func, ast.Name)
            and func.id == "print"
            and len(call.args) == 1
            and not call.keywords
        ):
            inner = call.args[0]
            if _is_named_call(inner, "read"):
                inner_source = _source(inner)
                if inner_source:
                    all_rewrites.append(
                        (start, end, [f"{indent}view{inner_source[len('read'):]}"])
                    )
                continue
            if _is_named_call(inner, *preview_targets):
                inner_source = _source(inner)
                if inner_source:
                    if _is_named_call(inner, "bash"):
                        var_name = _next_bash_var()
                        if _has_literal_bg_true(inner):
                            all_rewrites.append(
                                (start, end, [f"{indent}{var_name} = {inner_source}"])
                            )
                        else:
                            all_rewrites.append(
                                (start, end, [f"{indent}preview({var_name} := {inner_source})"])
                            )
                    else:
                        all_rewrites.append(
                            (start, end, [f"{indent}preview({inner_source})"])
                        )

    # Pattern 3: var = target(...) immediately followed by print(var)
    for i in range(len(body) - 1):
        assign = body[i]
        nxt = body[i + 1]

        if not isinstance(assign, ast.Assign):
            continue
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            continue
        if not isinstance(assign.value, ast.Call):
            continue
        afunc = assign.value.func
        if not (isinstance(afunc, ast.Name) and afunc.id in preview_targets):
            continue

        var_name = assign.targets[0].id

        if not isinstance(nxt, ast.Expr) or not isinstance(nxt.value, ast.Call):
            continue
        pcall = nxt.value
        if not (isinstance(pcall.func, ast.Name) and pcall.func.id == "print"):
            continue
        if len(pcall.args) != 1 or pcall.keywords:
            continue
        if not (isinstance(pcall.args[0], ast.Name) and pcall.args[0].id == var_name):
            continue

        pstart = nxt.lineno - 1
        pend = nxt.end_lineno - 1
        pindent = lines[pstart][: len(lines[pstart]) - len(lines[pstart].lstrip())]
        preview_uri_arg = _preview_uri_read_arg(assign.value)
        if preview_uri_arg is not None:
            all_rewrites.append(
                (pstart, pend, [f"{pindent}view({preview_uri_arg})"])
            )
        else:
            all_rewrites.append(
                (pstart, pend, [f"{pindent}preview({var_name})"])
            )

    if not all_rewrites:
        return code, counter

    all_rewrites.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_lines in all_rewrites:
        lines[start : end + 1] = new_lines

    return "\n".join(lines), counter