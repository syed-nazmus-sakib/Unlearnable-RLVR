#!/usr/bin/env python3
"""texguard -- refuse to write a macro file LaTeX cannot compile.

`crossmodel_numbers.tex` shipped with \\cmProsPredN40 and seven siblings. LaTeX
command names are sequences of CATEGORY-11 LETTERS ONLY: a digit in the name
ends the control sequence, so \\cmProsPredN40 parses as \\cmProsPredN followed
by the characters "40", and \\newcommand refuses to define it. It was the first
macro file in this project that would not build, and it reached the author.

Every emitter calls check() before writing. The failure is at emit time, in the
project, not at build time, in the paper.

  from texguard import check
  check(lines)                  # list of lines, or one string
  check(lines, path=out)        # names the file in the error

Also usable directly on files already written:

  python3 texguard.py analysis/*.tex
"""
import re
import sys

NEWCOMMAND = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\s*\{?\s*\\([^}\s\[{]+)")
LEGAL = re.compile(r"\A[A-Za-z]+\Z")


class TexGuard(Exception):
    pass


def offenders(text):
    """[(name, reason)] for every macro definition LaTeX would reject."""
    bad = []
    for name in NEWCOMMAND.findall(text):
        if LEGAL.match(name):
            continue
        if any(c.isdigit() for c in name):
            why = "contains a digit; LaTeX command names are letters only"
        elif "@" in name:
            why = "contains @, which is not a letter outside \\makeatletter"
        else:
            why = "contains a character that is not a letter"
        bad.append((name, why))
    return bad


def check(content, path=None):
    """Raise TexGuard if any macro name is illegal. Returns the macro count."""
    text = content if isinstance(content, str) else "\n".join(content)
    bad = offenders(text)
    if bad:
        where = f" in {path}" if path else ""
        lines = "\n".join(f"    \\{n}  -- {w}" for n, w in bad)
        raise TexGuard(
            f"{len(bad)} illegal macro name(s){where}; refusing to write:\n{lines}\n"
            f"  Spell numbers out: \\fooNForty, not \\fooN40.")
    return len(NEWCOMMAND.findall(text))


def main(argv):
    rc = 0
    for p in argv:
        try:
            n = check(open(p).read(), path=p)
            print(f"OK    {p}  ({n} macros)")
        except TexGuard as e:
            print(f"FAIL  {e}", file=sys.stderr)
            rc = 1
        except OSError as e:
            print(f"ERROR {p}: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) if len(sys.argv) > 1
             else (print(__doc__.split("Also usable")[0]) or 4))
