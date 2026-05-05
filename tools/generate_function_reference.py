from __future__ import annotations

import argparse
import ast
import html
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "function_reference.html"
SKIPPED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "compression_corpus",
    "zig-cache",
    "zig-out",
}
SOURCE_SUFFIXES = {".py", ".wgsl", ".metal", ".zig"}


@dataclass(frozen=True)
class ParameterDoc:
    name: str
    kind: str
    annotation: str
    default: str


@dataclass(frozen=True)
class FunctionDoc:
    name: str
    qualname: str
    module: str
    language: str
    kind: str
    signature: str
    file_path: Path
    line: int
    end_line: int
    doc: str
    parameters: tuple[ParameterDoc, ...]
    returns: str
    decorators: tuple[str, ...]


@dataclass(frozen=True)
class ParseIssue:
    file_path: Path
    message: str


def _relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _module_name_for_python(path: Path) -> str:
    rel = path.resolve().relative_to(PROJECT_ROOT)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _module_name_for_text(path: Path) -> str:
    return _relative_path(path)


def _iter_source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
            continue
        if any(part in SKIPPED_DIRS for part in path.relative_to(root).parts):
            continue
        yield path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _shorten(value: str, limit: int = 96) -> str:
    value = " ".join(value.strip().split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def _source_segment(source: str, node: ast.AST | None, *, fallback: str = "") -> str:
    if node is None:
        return fallback
    segment = ast.get_source_segment(source, node)
    if segment is None:
        try:
            segment = ast.unparse(node)
        except Exception:
            segment = fallback
    return _shorten(segment)


def _format_arg(arg: ast.arg, source: str, default: ast.AST | None = None, *, prefix: str = "") -> tuple[str, ParameterDoc]:
    annotation = _source_segment(source, arg.annotation)
    default_text = _source_segment(source, default) if default is not None else ""
    rendered = prefix + arg.arg
    if annotation:
        rendered += f": {annotation}"
    if default_text:
        rendered += f" = {default_text}"
    return rendered, ParameterDoc(arg.arg, "positional", annotation, default_text)


def _format_python_arguments(args: ast.arguments, source: str) -> tuple[list[str], list[ParameterDoc]]:
    rendered: list[str] = []
    parameters: list[ParameterDoc] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults: list[ast.AST | None] = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for index, arg in enumerate(args.posonlyargs):
        text, parameter = _format_arg(arg, source, defaults[index])
        rendered.append(text)
        parameters.append(ParameterDoc(parameter.name, "positional-only", parameter.annotation, parameter.default))

    if args.posonlyargs:
        rendered.append("/")

    for index, arg in enumerate(args.args, start=len(args.posonlyargs)):
        text, parameter = _format_arg(arg, source, defaults[index])
        rendered.append(text)
        parameters.append(parameter)

    if args.vararg is not None:
        text, parameter = _format_arg(args.vararg, source, prefix="*")
        rendered.append(text)
        parameters.append(ParameterDoc(parameter.name, "varargs", parameter.annotation, parameter.default))
    elif args.kwonlyargs:
        rendered.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text, parameter = _format_arg(arg, source, default)
        rendered.append(text)
        parameters.append(ParameterDoc(parameter.name, "keyword-only", parameter.annotation, parameter.default))

    if args.kwarg is not None:
        text, parameter = _format_arg(args.kwarg, source, prefix="**")
        rendered.append(text)
        parameters.append(ParameterDoc(parameter.name, "kwargs", parameter.annotation, parameter.default))

    return rendered, parameters


def _python_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, source: str) -> tuple[str, tuple[ParameterDoc, ...], str]:
    rendered_args, parameters = _format_python_arguments(node.args, source)
    returns = _source_segment(source, node.returns)
    prefix = "async fn" if isinstance(node, ast.AsyncFunctionDef) else "fn"
    signature = f"{prefix} {node.name}({', '.join(rendered_args)})"
    if returns:
        signature += f" -> {returns}"
    return signature, tuple(parameters), returns


def _decorator_text(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    decorators: list[str] = []
    for decorator in node.decorator_list:
        text = _source_segment(source, decorator)
        if text:
            decorators.append("@" + text)
    return tuple(decorators)


class _PythonFunctionVisitor(ast.NodeVisitor):
    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.module = _module_name_for_python(path)
        self.parents: list[tuple[str, str]] = []
        self.functions: list[FunctionDoc] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.parents.append(("class", node.name))
        self.generic_visit(node)
        self.parents.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        parent_names = [name for _, name in self.parents]
        qualname = ".".join(parent_names + [node.name])
        parent_kinds = {kind for kind, _ in self.parents}
        if "class" in parent_kinds:
            kind = "method"
        elif "function" in parent_kinds:
            kind = "nested function"
        else:
            kind = "function"

        signature, parameters, returns = _python_signature(node, self.source)
        self.functions.append(
            FunctionDoc(
                name=node.name,
                qualname=qualname,
                module=self.module,
                language="Python",
                kind=kind,
                signature=signature,
                file_path=self.path,
                line=int(node.lineno),
                end_line=int(getattr(node, "end_lineno", node.lineno) or node.lineno),
                doc=ast.get_docstring(node) or "",
                parameters=parameters,
                returns=returns,
                decorators=_decorator_text(self.source, node),
            )
        )
        self.parents.append(("function", node.name))
        self.generic_visit(node)
        self.parents.pop()


def _parse_python_file(path: Path) -> tuple[list[FunctionDoc], list[ParseIssue]]:
    try:
        source = _read_text(path)
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [], [ParseIssue(path, f"Python syntax error: {exc}")]
    visitor = _PythonFunctionVisitor(path, source)
    visitor.visit(tree)
    return visitor.functions, []


def _line_number_for_offset(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _end_line_for_segment(source: str, start: int, end: int) -> int:
    return _line_number_for_offset(source, end)


def _previous_comment_block(lines: list[str], function_line: int, *, skip_attributes: bool = False) -> tuple[str, tuple[str, ...]]:
    index = function_line - 2
    decorators: list[str] = []

    if skip_attributes:
        while index >= 0 and lines[index].strip().startswith("@"):
            decorators.append(lines[index].strip())
            index -= 1
        while index >= 0 and not lines[index].strip():
            index -= 1

    if index >= 0 and lines[index].strip().endswith("*/"):
        block: list[str] = []
        while index >= 0:
            block.append(lines[index].strip())
            if lines[index].strip().startswith("/**") or lines[index].strip().startswith("/*"):
                break
            index -= 1
        block.reverse()
        cleaned = []
        for line in block:
            line = line.removeprefix("/**").removeprefix("/*").removesuffix("*/").strip()
            line = line.removeprefix("*").strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned), tuple(reversed(decorators))

    comment_lines: list[str] = []
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            break
        if stripped.startswith("///") or stripped.startswith("//!") or stripped.startswith("//"):
            comment_lines.append(re.sub(r"^//[/!]?\s?", "", stripped))
        elif stripped.startswith("#"):
            comment_lines.append(stripped.removeprefix("#").strip())
        else:
            break
        index -= 1

    comment_lines.reverse()
    return "\n".join(line for line in comment_lines if line), tuple(reversed(decorators))


def _signature_until_body(source: str, start: int, search_from: int) -> tuple[str, int] | None:
    brace = source.find("{", search_from)
    semicolon = source.find(";", search_from)
    if brace == -1:
        return None
    if semicolon != -1 and semicolon < brace:
        return None
    signature = source[start:brace].strip()
    return _shorten(signature, limit=240), brace


def _parse_wgsl_file(path: Path) -> list[FunctionDoc]:
    source = _read_text(path)
    lines = source.splitlines()
    docs: list[FunctionDoc] = []
    pattern = re.compile(r"(?m)^\s*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for match in pattern.finditer(source):
        signature_info = _signature_until_body(source, match.start(), match.end())
        if signature_info is None:
            continue
        signature, body_start = signature_info
        line = _line_number_for_offset(source, match.start())
        doc, decorators = _previous_comment_block(lines, line, skip_attributes=True)
        docs.append(
            FunctionDoc(
                name=match.group(1),
                qualname=match.group(1),
                module=_module_name_for_text(path),
                language="WGSL",
                kind="shader function",
                signature=signature,
                file_path=path,
                line=line,
                end_line=_end_line_for_segment(source, match.start(), body_start),
                doc=doc,
                parameters=(),
                returns="",
                decorators=decorators,
            )
        )
    return docs


def _parse_zig_file(path: Path) -> list[FunctionDoc]:
    source = _read_text(path)
    lines = source.splitlines()
    docs: list[FunctionDoc] = []
    pattern = re.compile(r"(?m)^\s*(?:(?:pub|export)\s+)*fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for match in pattern.finditer(source):
        signature_info = _signature_until_body(source, match.start(), match.end())
        if signature_info is None:
            continue
        signature, body_start = signature_info
        line = _line_number_for_offset(source, match.start())
        doc, decorators = _previous_comment_block(lines, line)
        docs.append(
            FunctionDoc(
                name=match.group(1),
                qualname=match.group(1),
                module=_module_name_for_text(path),
                language="Zig",
                kind="native function",
                signature=signature,
                file_path=path,
                line=line,
                end_line=_end_line_for_segment(source, match.start(), body_start),
                doc=doc,
                parameters=(),
                returns="",
                decorators=decorators,
            )
        )
    return docs


def _parse_metal_file(path: Path) -> list[FunctionDoc]:
    source = _read_text(path)
    lines = source.splitlines()
    docs: list[FunctionDoc] = []
    pattern = re.compile(
        r"(?m)^\s*(?:(?:static|inline|constexpr|constant|kernel|vertex|fragment)\s+)*"
        r"[A-Za-z_][A-Za-z0-9_:<>,\s*&]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    )
    for match in pattern.finditer(source):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "return"}:
            continue
        signature_info = _signature_until_body(source, match.start(), match.end())
        if signature_info is None:
            continue
        signature, body_start = signature_info
        line = _line_number_for_offset(source, match.start())
        doc, decorators = _previous_comment_block(lines, line)
        docs.append(
            FunctionDoc(
                name=name,
                qualname=name,
                module=_module_name_for_text(path),
                language="Metal",
                kind="shader function",
                signature=signature,
                file_path=path,
                line=line,
                end_line=_end_line_for_segment(source, match.start(), body_start),
                doc=doc,
                parameters=(),
                returns="",
                decorators=decorators,
            )
        )
    return docs


def collect_function_docs(root: Path) -> tuple[list[FunctionDoc], list[ParseIssue]]:
    functions: list[FunctionDoc] = []
    issues: list[ParseIssue] = []

    for path in _iter_source_files(root):
        if path.suffix == ".py":
            parsed, parsed_issues = _parse_python_file(path)
            functions.extend(parsed)
            issues.extend(parsed_issues)
        elif path.suffix == ".wgsl":
            functions.extend(_parse_wgsl_file(path))
        elif path.suffix == ".metal":
            functions.extend(_parse_metal_file(path))
        elif path.suffix == ".zig":
            functions.extend(_parse_zig_file(path))

    functions.sort(key=lambda doc: (_relative_path(doc.file_path), doc.line, doc.qualname))
    return functions, issues


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-").lower()
    return slug or "item"


def _function_id(doc: FunctionDoc) -> str:
    return f"fn-{_slug(doc.module)}-{doc.line}-{_slug(doc.qualname)}"


def _module_id(module: str) -> str:
    return f"module-{_slug(module)}"


def _escape(value: str) -> str:
    return html.escape(value, quote=True)


def _render_doc(doc: FunctionDoc) -> str:
    if not doc.doc.strip():
        return '<p class="missing-doc">No source docstring or doc comment was found for this item.</p>'
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", doc.doc.strip()) if part.strip()]
    rendered = []
    for paragraph in paragraphs:
        if "\n" in paragraph:
            rendered.append(f'<pre class="doc-block">{_escape(paragraph)}</pre>')
        else:
            rendered.append(f"<p>{_escape(paragraph)}</p>")
    return "\n".join(rendered)


def _render_decorators(doc: FunctionDoc) -> str:
    if not doc.decorators:
        return ""
    chips = "\n".join(f'<code class="decorator">{_escape(item)}</code>' for item in doc.decorators)
    return f'<div class="decorators">{chips}</div>'


def _render_parameters(doc: FunctionDoc) -> str:
    if not doc.parameters:
        return ""
    rows = []
    for param in doc.parameters:
        annotation = (
            f"<code>{_escape(param.annotation)}</code>"
            if param.annotation
            else '<span class="muted">inferred</span>'
        )
        default = f"<code>{_escape(param.default)}</code>" if param.default else '<span class="muted">required</span>'
        rows.append(
            "<tr>"
            f"<td><code>{_escape(param.name)}</code></td>"
            f"<td>{_escape(param.kind)}</td>"
            f"<td>{annotation}</td>"
            f"<td>{default}</td>"
            "</tr>"
        )
    return (
        '<h4>Parameters</h4>'
        '<table class="params">'
        "<thead><tr><th>Name</th><th>Kind</th><th>Type</th><th>Default</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _source_href(output_path: Path, source_path: Path) -> str:
    rel = os.path.relpath(source_path, output_path.parent)
    return quote(rel.replace(os.sep, "/"), safe="/._-")


def _render_function(doc: FunctionDoc, output_path: Path) -> str:
    source = _relative_path(doc.file_path)
    source_href = _source_href(output_path, doc.file_path)
    returns = f'<span class="meta-pill">returns <code>{_escape(doc.returns)}</code></span>' if doc.returns else ""
    tags = (
        f'<span class="meta-pill">{_escape(doc.language)}</span>'
        f'<span class="meta-pill">{_escape(doc.kind)}</span>'
        f"{returns}"
    )
    return (
        f'<article class="item" id="{_function_id(doc)}" data-name="{_escape(doc.module + " " + doc.qualname + " " + source)}">'
        f'<h3><span class="kind">fn</span> <code>{_escape(doc.qualname)}</code></h3>'
        f'<pre class="signature"><code>{_escape(doc.signature)}</code></pre>'
        f"{_render_decorators(doc)}"
        f'<div class="item-meta">{tags}<a href="{source_href}" class="source-link">{_escape(source)}:{doc.line}</a></div>'
        f'<div class="doc">{_render_doc(doc)}</div>'
        f"{_render_parameters(doc)}"
        "</article>"
    )


def _render_sidebar(grouped: dict[str, list[FunctionDoc]], counts: Counter[str]) -> str:
    module_links = []
    for module, docs in grouped.items():
        label = _escape(module)
        module_links.append(
            f'<a href="#{_module_id(module)}"><span>{label}</span><span class="count">{len(docs)}</span></a>'
        )
    language_counts = "\n".join(
        f'<li><span>{_escape(language)}</span><strong>{count}</strong></li>'
        for language, count in sorted(counts.items())
    )
    return (
        '<aside class="sidebar">'
        '<div class="crate">Minechunk</div>'
        '<label class="search-label" for="search">Search functions</label>'
        '<input id="search" type="search" placeholder="module, name, or file" autocomplete="off">'
        f'<ul class="language-counts">{language_counts}</ul>'
        '<nav class="module-nav">'
        f"{''.join(module_links)}"
        "</nav>"
        "</aside>"
    )


def render_html(functions: list[FunctionDoc], issues: list[ParseIssue], output_path: Path) -> str:
    grouped: dict[str, list[FunctionDoc]] = defaultdict(list)
    for doc in functions:
        grouped[doc.module].append(doc)
    grouped = dict(sorted(grouped.items(), key=lambda item: item[0]))
    counts = Counter(doc.language for doc in functions)
    documented = sum(1 for doc in functions if doc.doc.strip())

    module_sections = []
    for module, docs in grouped.items():
        items = "\n".join(_render_function(doc, output_path) for doc in docs)
        module_sections.append(
            f'<section class="module-section" id="{_module_id(module)}" data-module-section>'
            f'<h2>Module <code>{_escape(module)}</code> <span>{len(docs)} functions</span></h2>'
            f"{items}"
            "</section>"
        )

    issue_html = ""
    if issues:
        issue_items = "".join(
            f"<li><code>{_escape(_relative_path(issue.file_path))}</code>: {_escape(issue.message)}</li>" for issue in issues
        )
        issue_html = f'<section class="issues"><h2>Parse Issues</h2><ul>{issue_items}</ul></section>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Minechunk Function Reference</title>
<style>
:root {{
  color-scheme: light;
  --bg: #ffffff;
  --panel: #f6f8fa;
  --panel-strong: #eef1f4;
  --border: #d9dee3;
  --text: #1f2328;
  --muted: #66707a;
  --link: #0969da;
  --code: #f3f4f6;
  --accent: #a15c00;
  --accent-soft: #fff4df;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: 0;
}}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
code, pre {{
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.92em;
}}
.layout {{
  display: grid;
  grid-template-columns: minmax(240px, 310px) minmax(0, 1fr);
  min-height: 100vh;
}}
.sidebar {{
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
  background: var(--panel);
  border-right: 1px solid var(--border);
  padding: 18px 14px;
}}
.crate {{
  font-size: 1.3rem;
  font-weight: 700;
  margin-bottom: 16px;
}}
.search-label {{
  display: block;
  color: var(--muted);
  font-size: 0.8rem;
  font-weight: 650;
  margin-bottom: 6px;
}}
#search {{
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  color: var(--text);
  background: #fff;
}}
.language-counts {{
  display: grid;
  gap: 4px;
  list-style: none;
  padding: 10px 0 12px;
  margin: 12px 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}}
.language-counts li {{
  display: flex;
  justify-content: space-between;
  color: var(--muted);
  font-size: 0.88rem;
}}
.language-counts strong {{ color: var(--text); }}
.module-nav {{
  display: grid;
  gap: 2px;
}}
.module-nav a {{
  display: flex;
  gap: 8px;
  align-items: baseline;
  justify-content: space-between;
  min-width: 0;
  padding: 5px 6px;
  border-radius: 6px;
  color: var(--text);
}}
.module-nav a:hover {{
  background: var(--panel-strong);
  text-decoration: none;
}}
.module-nav span:first-child {{
  overflow-wrap: anywhere;
}}
.count {{
  flex: none;
  color: var(--muted);
  font-size: 0.8rem;
}}
main {{
  min-width: 0;
  padding: 30px clamp(18px, 4vw, 56px) 56px;
}}
.hero {{
  max-width: 980px;
  margin-bottom: 28px;
}}
h1 {{
  margin: 0 0 10px;
  font-size: clamp(2rem, 4vw, 3.4rem);
  line-height: 1.08;
}}
.hero p {{
  max-width: 760px;
  color: var(--muted);
  margin: 0;
}}
.stats {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}}
.stat {{
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--panel);
  padding: 8px 10px;
  min-width: 124px;
}}
.stat strong {{
  display: block;
  font-size: 1.1rem;
}}
.stat span {{
  color: var(--muted);
  font-size: 0.82rem;
}}
.module-section {{
  max-width: 1120px;
  margin-top: 28px;
}}
.module-section > h2 {{
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
  margin: 0 0 14px;
  font-size: 1.35rem;
}}
.module-section > h2 span {{
  color: var(--muted);
  font-size: 0.85rem;
  font-weight: 500;
}}
.item {{
  border: 1px solid var(--border);
  border-radius: 6px;
  margin: 12px 0;
  padding: 14px;
  background: #fff;
}}
.item h3 {{
  display: flex;
  align-items: baseline;
  gap: 8px;
  margin: 0 0 8px;
  font-size: 1.05rem;
  overflow-wrap: anywhere;
}}
.kind {{
  color: var(--accent);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-weight: 700;
}}
.signature {{
  margin: 0 0 10px;
  padding: 10px 12px;
  overflow-x: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  background: var(--code);
  border-radius: 6px;
  border: 1px solid var(--border);
}}
.decorators {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 9px;
}}
.decorator {{
  color: #5a32a3;
  background: #f3edff;
  border: 1px solid #e2d7fb;
  border-radius: 999px;
  padding: 2px 7px;
}}
.item-meta {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 7px;
  margin-bottom: 10px;
}}
.meta-pill {{
  display: inline-flex;
  gap: 4px;
  align-items: center;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--accent-soft);
  color: #684100;
  padding: 2px 8px;
  font-size: 0.82rem;
}}
.source-link {{
  font-size: 0.86rem;
}}
.doc p {{
  margin: 0 0 10px;
}}
.missing-doc {{
  color: var(--muted);
  font-style: italic;
}}
.doc-block {{
  margin: 0 0 10px;
  padding: 10px;
  background: var(--panel);
  border-radius: 6px;
  border: 1px solid var(--border);
  white-space: pre-wrap;
}}
h4 {{
  margin: 12px 0 6px;
  font-size: 0.9rem;
}}
.params {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}}
.params th,
.params td {{
  border: 1px solid var(--border);
  padding: 6px 8px;
  text-align: left;
  vertical-align: top;
}}
.params th {{
  background: var(--panel);
}}
.muted {{ color: var(--muted); }}
.issues {{
  max-width: 980px;
  border: 1px solid #f0c36d;
  border-radius: 6px;
  padding: 12px 16px;
  background: #fff8e6;
}}
.hidden-by-search {{ display: none; }}
@media (max-width: 820px) {{
  .layout {{
    display: block;
  }}
  .sidebar {{
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }}
  .module-nav {{
    max-height: 240px;
    overflow: auto;
  }}
  main {{
    padding: 22px 14px 40px;
  }}
}}
</style>
</head>
<body>
<div class="layout">
{_render_sidebar(grouped, counts)}
<main>
  <section class="hero">
    <h1>Function Reference</h1>
    <p>Rustdoc-style reference generated from Minechunk source files. Python entries come from the AST; WGSL, Metal, and Zig entries use source signature extraction.</p>
    <div class="stats">
      <div class="stat"><strong>{len(functions)}</strong><span>functions</span></div>
      <div class="stat"><strong>{len(grouped)}</strong><span>modules and source files</span></div>
      <div class="stat"><strong>{documented}</strong><span>with doc text</span></div>
      <div class="stat"><strong>{len(issues)}</strong><span>parse issues</span></div>
    </div>
  </section>
  {issue_html}
  {''.join(module_sections)}
</main>
</div>
<script>
const search = document.getElementById("search");
const items = Array.from(document.querySelectorAll(".item"));
const modules = Array.from(document.querySelectorAll("[data-module-section]"));
function applyFilter() {{
  const query = search.value.trim().toLowerCase();
  for (const item of items) {{
    const haystack = item.dataset.name.toLowerCase();
    item.classList.toggle("hidden-by-search", query.length > 0 && !haystack.includes(query));
  }}
  for (const section of modules) {{
    const visible = Array.from(section.querySelectorAll(".item")).some((item) => !item.classList.contains("hidden-by-search"));
    section.classList.toggle("hidden-by-search", !visible);
  }}
}}
search.addEventListener("input", applyFilter);
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Rustdoc-style HTML function reference.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="HTML file to write. Defaults to docs/function_reference.html.",
    )
    args = parser.parse_args(argv)

    output_path = args.output
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    functions, issues = collect_function_docs(PROJECT_ROOT)
    output_path.write_text(render_html(functions, issues, output_path), encoding="utf-8")
    print(
        f"wrote {_relative_path(output_path)} "
        f"with {len(functions)} functions across {len({doc.module for doc in functions})} modules"
    )
    if issues:
        print(f"encountered {len(issues)} parse issues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
