#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the guarantee records in spec/ and render docs/specification.md from them.

    tools/spec.py check            the schema, the cross-references, and every quote against its source
    tools/spec.py check --offline  the same without fetching sources (quotes are not checked)
    tools/spec.py render           write docs/specification.md
    tools/spec.py render --check   fail if docs/specification.md is not what render would write

Python 3.11 or later, standard library only.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import tomllib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = ROOT / "spec"
OUT = ROOT / "docs" / "specification.md"

KINDS = {"forbids", "permits"}
BASES = {"documented", "read-from-source", "observed-on-hardware", "assumed"}
STATUSES = {
    "conforms": "does what the record says",
    "hides-bugs": "code that is wrong on the hardware can pass here",
    "invents-bugs": "code that is right on the hardware can fail here",
    "not-implemented": "the provider does not offer this",
    "unverified": "not yet established",
}
REQUIRED = ["id", "title", "kind", "statement", "scope", "basis", "sources", "checked_by", "srdsim"]
ID_RE = re.compile(r"^[A-Z]+-[A-Z]+-\d{3}$")


def load():
    sources = tomllib.loads((SPEC / "sources.toml").read_text())["sources"]
    records = []
    for path in sorted(SPEC.glob("*.toml")):
        if path.name == "sources.toml":
            continue
        for r in tomllib.loads(path.read_text()).get("guarantee", []):
            r["_file"] = path.name
            records.append(r)
    return sources, records


def ctest_names():
    """Test names CMakeLists.txt registers, with its foreach lists expanded."""
    text = (ROOT / "CMakeLists.txt").read_text()
    names = set(re.findall(r"add_test\(NAME ([A-Za-z0-9_]+)\b", text))
    lists = dict(re.findall(r"set\((SRDSIM_HOST_TESTS)\s+([^)]*)\)", text))
    for var, items in re.findall(r"foreach\((\w+) IN (?:ITEMS|LISTS) ([^)]*)\)", text):
        values = lists.get(items.strip(), items).split()
        for template in re.findall(r"add_test\(NAME ([A-Za-z0-9_]*\$\{%s\}[A-Za-z0-9_]*)" % var, text):
            for v in values:
                names.add(template.replace("${%s}" % var, v))
    return names


def normalise(text: str) -> str:
    """Compare quotes and sources without Markdown emphasis, line breaks or repeated spaces."""
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "").replace("\\\n", " ")).strip()


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def check(offline: bool) -> int:
    sources, records = load()
    errors, notes = [], []
    tests = ctest_names()
    seen = {}
    for r in records:
        where = f"{r.get('_file')}:{r.get('id', '?')}"
        for field in REQUIRED:
            if field not in r:
                errors.append(f"{where}: missing field {field}")
        if not ID_RE.match(r.get("id", "")):
            errors.append(f"{where}: id must look like EFA-RMA-001")
        if r.get("id") in seen:
            errors.append(f"{where}: id also used in {seen[r['id']]}")
        seen[r.get("id")] = r.get("_file")
        if r.get("kind") not in KINDS:
            errors.append(f"{where}: kind must be one of {sorted(KINDS)}")
        if r.get("basis") not in BASES:
            errors.append(f"{where}: basis must be one of {sorted(BASES)}")
        if r.get("basis") == "assumed" and not r.get("basis_note"):
            errors.append(f"{where}: an assumed guarantee says in basis_note what it rests on")
        if r.get("basis") != "assumed" and not r.get("sources"):
            errors.append(f"{where}: only an assumed guarantee may have no source")
        status = r.get("srdsim", {}).get("status")
        if status not in STATUSES:
            errors.append(f"{where}: srdsim.status must be one of {sorted(STATUSES)}")
        if not r.get("srdsim", {}).get("note"):
            errors.append(f"{where}: srdsim.note says why the status is what it is")
        if not r.get("checked_by") and not r.get("unchecked"):
            errors.append(f"{where}: nothing checks it, and unchecked does not say why")
        for ref in r.get("checked_by", []):
            kind, _, name = ref.partition(":")
            if kind == "test":
                if name not in tests:
                    errors.append(f"{where}: checked_by names {ref}, which CMakeLists.txt does not register")
            elif kind in ("model", "fact"):
                notes.append(f"{where}: {ref} is not resolved yet")
            else:
                errors.append(f"{where}: checked_by entries are test:, model: or fact:, not {ref}")
        for s in r.get("sources", []):
            if s.get("source") not in sources:
                errors.append(f"{where}: unknown source {s.get('source')}")
            if not s.get("quote"):
                errors.append(f"{where}: a source entry quotes the text it relies on")

    if not offline:
        cache = {}
        for r in records:
            for s in r.get("sources", []):
                src = sources.get(s.get("source"))
                if not src:
                    continue
                if s["source"] not in cache:
                    cache[s["source"]] = normalise(fetch(src["url"]))
                if normalise(s["quote"]) not in cache[s["source"]]:
                    errors.append(f"{r['_file']}:{r['id']}: quote not found in {s['source']} ({src['version']}): "
                                  f"{s['quote'][:80]}")

    by_status = {}
    for r in records:
        by_status.setdefault(r.get("srdsim", {}).get("status"), []).append(r.get("id"))
    unchecked = [r["id"] for r in records if not r.get("checked_by")]
    quotes = "quotes not checked (offline)" if offline else \
        ("every quote found in its source" if not any("quote not found" in e for e in errors) else "SOME QUOTES NOT FOUND")
    print(f"{len(records)} guarantees in {len({r['_file'] for r in records})} files, {len(sources)} sources, {quotes}")
    for st in STATUSES:
        if by_status.get(st):
            print(f"  {st:16} {len(by_status[st]):2}  {' '.join(by_status[st])}")
    print(f"  nothing checks {len(unchecked)}: {' '.join(unchecked)}")
    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


def render() -> str:
    sources, records = load()
    out = [
        "<!-- Generated by tools/spec.py render from spec/*.toml. Do not edit by hand. -->",
        "# What the fabric guarantees",
        "",
        "Each row is one record in [`spec/`](../spec/). A record either **forbids** something (the fabric "
        "never does it, so this project must not either) or **permits** it (the fabric may do it, so this "
        "project should be able to). The last column says how this project compares: `hides-bugs` means code that is "
        "wrong on the hardware can pass here, and `invents-bugs` that code that is right on the hardware can fail here.",
        "",
        "| ID | Guarantee | Kind | Basis | Checked by | This project |",
        "|---|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda r: r["id"]):
        checks = ", ".join(f"`{c.split(':', 1)[1]}`" for c in r["checked_by"]) or "nothing yet"
        out.append(f"| {r['id']} | {r['statement']} | {r['kind']} | {r['basis']} | {checks} | {r['srdsim']['status']} |")
    out += ["", "## Known differences", "",
            "The records where this project does not do what the fabric does, with the reason.", ""]
    for r in sorted(records, key=lambda r: r["id"]):
        st = r["srdsim"]["status"]
        if st != "conforms":
            out.append(f"- **{r['id']}** ({st}: {STATUSES[st]}). {r['srdsim']['note']}")
    out += ["", "## Sources", ""]
    for key, src in sources.items():
        out.append(f"- `{key}`: [{src['title']}]({src['view']}), {src['version']}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "render"])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.command == "check":
        return check(a.offline)
    text = render()
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run tools/spec.py render")
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
