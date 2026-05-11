import re


PREVIEW_URI_PREFIX = "session://preview/"
_PREVIEW_REF_RE = re.compile(
    r"\[PreviewRef: (?P<uri>session://preview/[^\]\n]+)\]\n"
    r".*?"
    r"\[/PreviewRef\]",
    re.DOTALL,
)


def is_preview_uri(value) -> bool:
    return isinstance(value, str) and value.startswith(PREVIEW_URI_PREFIX)


def preview_key(uri: str) -> str:
    if not is_preview_uri(uri):
        raise ValueError(f"Not a preview URI: {uri}")
    key = uri[len(PREVIEW_URI_PREFIX):]
    if not key:
        raise ValueError(f"Not a preview URI: {uri}")
    return key


def numbered_content(content: str) -> str:
    return "\n".join(f"{i+1:>5}→{line}" for i, line in enumerate(content.split("\n")))


def render_preview_refs(content: str, expanded_refs: dict, load_preview) -> str:
    if not content or not expanded_refs:
        return content

    def replace(match):
        uri = match.group("uri")
        options = expanded_refs.get(uri)
        if not options:
            return match.group(0)
        full = load_preview(uri)
        if options.get("numbered"):
            full = numbered_content(full)
        return f"[ExpandedPreviewRef: {uri}]\n{full}\n[/ExpandedPreviewRef]"

    return _PREVIEW_REF_RE.sub(replace, content)
