# SPDX-License-Identifier: Apache-2.0
"""Write every figure in English (NAME.svg) and Japanese (NAME.ja.svg) next to this directory.

    python3 docs/images/src/build.py            # all figures
    python3 docs/images/src/build.py modes      # one figure

Fails, listing each label, when a label would overflow its box or the canvas.
"""
import importlib.util
import pathlib
import sys

from svgkit import Lang

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent


def main():
    only = set(sys.argv[1:])
    problems = []
    for src in sorted(HERE.glob("*.py")):
        if src.name in ("svgkit.py", "build.py") or (only and src.stem not in only):
            continue
        spec = importlib.util.spec_from_file_location(src.stem, src)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for code, suffix in (("en", ""), ("ja", ".ja")):
            fig = mod.draw(Lang(code))
            (OUT / f"{mod.NAME}{suffix}.svg").write_text(fig.svg())
            problems += [f"{mod.NAME}{suffix}.svg: {p}" for p in fig.problems]
    for p in problems:
        print(p, file=sys.stderr)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
