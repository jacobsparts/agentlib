"""CodeAgent-specific preprocessing: rewrite bare tool calls to assignment + preview."""

import ast

_INVALID_VIEW_MESSAGE = "view() is a display tool, not a value. Use read() for file contents as text."


def preprocess_code_agent(
    code: str,
    *,
    preview_targets: frozenset[str],
    preview_counter: int = 0,
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

    # Pattern 1: bare target call or print(target(...))
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

    # Pattern 2: var = target(...) immediately followed by print(var)
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
        all_rewrites.append(
            (pstart, pend, [f"{pindent}preview({var_name})"])
        )

    if not all_rewrites:
        return code, counter

    all_rewrites.sort(key=lambda x: x[0], reverse=True)
    for start, end, new_lines in all_rewrites:
        lines[start : end + 1] = new_lines

    return "\n".join(lines), counter
