"""The console is transpiled in the browser, so nothing else catches a typo.

`templates/console.html` pulls React + Babel from a CDN and hands it seven
in-repo files, each compiled at page load. A renamed API method, a missing
`<script>` target, or a component that no file defines produces a blank page
and a console error -- and every Python test still passes.

These checks are static: they read the files and cross-reference the names.
No node, no browser, no network -- so they run wherever pytest runs.

They are a floor, not a parser. To actually compile the JSX with the same
Babel the browser uses::

    curl -sLo /tmp/babel.js https://unpkg.com/@babel/standalone@7.29.0/babel.min.js
    node -e 'const B=require("/tmp/babel.js"),fs=require("fs");
      for (const f of fs.readdirSync("static/console").filter(f=>f.endsWith(".jsx")))
        B.transform(fs.readFileSync("static/console/"+f,"utf8"),
                    {presets:["react"],filename:f});
      console.log("all jsx compiles");'

(There is no CI in this repo, so nothing runs that for you.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CONSOLE = _ROOT / "static" / "console"
_TEMPLATE = _ROOT / "templates" / "console.html"

_JSX_FILES = sorted(_CONSOLE.glob("*.jsx"))


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_console_template_exists():
    assert _TEMPLATE.is_file()


def test_every_static_asset_the_template_requests_exists():
    """`url_for('static', filename=...)` resolving to nothing is a 404 at load."""
    html = _read(_TEMPLATE)
    referenced = re.findall(r"filename='([^']+)'", html)
    assert referenced, "template requests no static assets -- did the syntax change?"
    missing = [f for f in referenced if not (_ROOT / "static" / f).is_file()]
    assert not missing, f"console.html references missing static files: {missing}"


def test_every_jsx_file_on_disk_is_loaded_by_the_template():
    """A component nobody loads is dead weight; more likely, a forgotten script tag."""
    html = _read(_TEMPLATE)
    for jsx in _JSX_FILES:
        assert f"console/{jsx.name}" in html, f"{jsx.name} is never loaded by console.html"


def test_api_js_exports_every_method_the_components_call():
    """`API.foo(...)` where api.js exports no `foo` is a TypeError at runtime."""
    api_src = _read(_CONSOLE / "api.js")
    export_block = re.search(
        r"window\.MonadLabsAPI\s*=\s*\{(.*?)\}\s*;", api_src, re.S
    )
    assert export_block, "could not find the window.MonadLabsAPI export block"
    exported = set(re.findall(r"[A-Za-z_]\w*", export_block.group(1)))

    used: set[str] = set()
    for jsx in _JSX_FILES:
        used.update(re.findall(r"\bAPI\.([A-Za-z_]\w*)", _read(jsx)))
    assert used, "no API calls found -- did the components stop using the API?"

    missing = sorted(used - exported)
    assert not missing, f"components call API methods api.js does not export: {missing}"


def test_every_component_global_used_is_defined_somewhere():
    """Components hand each other `window.ConsoleX`; a rename breaks the tree."""
    sources = {jsx.name: _read(jsx) for jsx in _JSX_FILES}
    defined: set[str] = set()
    for src in sources.values():
        defined.update(re.findall(r"window\.(Console\w+)\s*=", src))

    used: set[str] = set()
    for src in sources.values():
        used.update(re.findall(r"window\.(Console\w+)\b", src))

    missing = sorted(used - defined)
    assert not missing, f"components reference undefined globals: {missing}"


def test_design_system_global_matches_the_bundle():
    """The vendored bundle keeps its generated namespace; consumers must agree."""
    ds = _read(_ROOT / "static" / "ds" / "ds-primitives.js")
    names = set(re.findall(r"window\.(ApertureDesignSystem_\w+)\s*=", ds))
    assert len(names) == 1, f"expected exactly one DS global, found {names}"
    (ds_global,) = names

    for consumer in [*_JSX_FILES, _TEMPLATE]:
        for used in re.findall(r"(ApertureDesignSystem_\w+)", _read(consumer)):
            assert used == ds_global, (
                f"{consumer.name} uses {used}, but the bundle defines {ds_global}"
            )


_DECL = re.compile(r"\b(?:const|let|var|function|class)\s+([A-Za-z_$][\w$]*)")
_DESTRUCTURE = re.compile(r"\b(?:const|let|var)\s*\{([^}]*)\}\s*=")


def _global_scope_names(src: str) -> set[str]:
    """Names a file introduces into the *shared global* lexical scope.

    Scope, not indentation: a declaration is global only at brace depth 0.
    Anything inside the file's IIFE is private no matter what column it sits
    in, so this must track depth rather than match `^const`.
    """
    code = _strip_comments_and_strings(src)
    names: set[str] = set()
    depth = 0
    for line in code.splitlines():
        if depth == 0:
            names.update(_DECL.findall(line))
            for group in _DESTRUCTURE.findall(line):
                for part in group.split(","):
                    # `{ Tabs, Button, Badge }` and `{ a: b }` -> bound name.
                    bound = part.split(":")[-1].strip()
                    if re.fullmatch(r"[A-Za-z_$][\w$]*", bound):
                        names.add(bound)
        depth += line.count("{") - line.count("}")
    return names


def test_no_two_jsx_files_declare_the_same_top_level_name():
    """Every `type="text/babel"` script shares ONE global lexical scope.

    Two files declaring the same top-level `const`/`function` is a
    SyntaxError ("Identifier 'X' has already been declared") that aborts the
    *entire* later file. When that file is app.jsx, `ReactDOM.createRoot`
    never runs and the page is blank -- which is precisely what shipped:
    app.jsx's `const IconRail` collided with IconRail.jsx's
    `function IconRail`, and CopilotChat.jsx's `const { Badge }` collided
    with ResultViewer.jsx's.

    The fix is that each file wraps itself in an IIFE, so this test asserts
    the collision is impossible rather than merely absent today.
    """
    owners: dict[str, list[str]] = {}
    for jsx in _JSX_FILES:
        for name in _global_scope_names(_read(jsx)):
            owners.setdefault(name, []).append(jsx.name)

    clashes = {n: sorted(f) for n, f in owners.items() if len(f) > 1}
    assert not clashes, (
        "these names are declared at top level in more than one JSX file; "
        f"the browser aborts the later file with a SyntaxError: {clashes}"
    )


@pytest.mark.parametrize("jsx", _JSX_FILES, ids=lambda p: p.name)
def test_each_jsx_file_keeps_its_locals_out_of_the_global_scope(jsx: Path):
    """Each console file must wrap its body in an IIFE.

    Without it, the file's top-level names leak into the scope every other
    `text/babel` script shares, and the next file that picks the same name
    silently kills the page. Publishing happens through `window.Console*`.
    """
    code = _strip_comments_and_strings(_read(jsx))
    assert re.search(r"^\(function\s*\(\s*\)\s*\{", code, re.M), (
        f"{jsx.name} does not open with an IIFE; its top-level declarations "
        "leak into the shared global scope"
    )
    assert re.search(r"^\}\)\(\);", code, re.M), f"{jsx.name} does not close its IIFE"


def _strip_comments_and_strings(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"'(?:\\.|[^'\\])*'", "''", src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', '""', src)
    src = re.sub(r"`(?:\\.|[^`\\])*`", "``", src)
    return src


def test_no_live_references_to_the_retired_prototype_helpers():
    """The design prototype shipped fake data; wiring it to the API removed it.

    Comments *about* the removed helpers are fine and in fact desirable -- the
    components explain what `pickFakeFiles` used to do and why it is gone. Only
    live code counts, so comments and string literals are stripped first.
    """
    retired = ["pickFakeFiles", "ChatsPanel", "MONADLABS_FAKE"]
    for jsx in _JSX_FILES:
        code = _strip_comments_and_strings(_read(jsx))
        for name in retired:
            assert name not in code, f"{jsx.name} still calls {name} in live code"


def test_no_references_to_the_deleted_dev_tool_frontend():
    html = _read(_TEMPLATE)
    for gone in ("static/app.js", "static/style.css", "templates/index.html"):
        assert gone not in html


@pytest.mark.parametrize("jsx", _JSX_FILES, ids=lambda p: p.name)
def test_jsx_braces_and_parens_balance(jsx: Path):
    """A crude parse: catches the truncated-file and stray-brace cases.

    Not a substitute for Babel, but it runs everywhere and would have caught
    a half-written component.
    """
    # Strip strings and comments so braces inside them don't count.
    src = _strip_comments_and_strings(_read(jsx))
    for open_c, close_c in (("{", "}"), ("(", ")"), ("[", "]")):
        assert src.count(open_c) == src.count(close_c), (
            f"{jsx.name}: unbalanced {open_c}{close_c}"
        )


def test_api_calls_are_same_origin():
    """A credentials-carrying fetch to another origin would leak the session."""
    api_src = _read(_CONSOLE / "api.js")
    fetches = re.findall(r"fetch\(\s*([^,)]+)", api_src)
    assert fetches, "no fetch calls found in api.js"
    for target in fetches:
        assert "http://" not in target and "https://" not in target, (
            f"api.js fetches an absolute URL: {target}"
        )
    assert '"same-origin"' in api_src or "'same-origin'" in api_src
