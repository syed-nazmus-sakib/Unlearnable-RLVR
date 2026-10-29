#!/usr/bin/env python3
"""procfind -- the process check that cannot match its own caller.

Replaces every use of `pgrep -f <pattern>` in this project. `pgrep -f` matches
against full command lines, which includes the command line of the shell that
is running the check, so a pattern naming the target also names the caller.
That returned the caller's PID three times here; the third time `kill -9` on
the result killed the invoking shell (A59, A61.6).

CONTRACT
  * candidates come from /proc/<pid>/cmdline, never from a process listing that
    includes this program's own argv;
  * this process and EVERY ancestor of it are excluded by PID;
  * if the pattern matches this process or any ancestor, nothing is returned
    at all -- the check is refused, because a pattern that describes the caller
    cannot distinguish caller from target;
  * output is bare PIDs, one per line, safe to pass to kill.

EXIT CODES
  0  one or more matches, printed
  1  no match (not an error: nothing is running)
  3  REFUSED -- the pattern matches the caller or an ancestor
  4  bad usage

USAGE
  procfind.py <pattern>              substring match against the full cmdline
  procfind.py --re <regex>           python regex against the full cmdline
  procfind.py --exe <name>           exact basename match on argv[0] only
  procfind.py --quiet <pattern>      print nothing, exit code only

  pids=$(procfind.py 'checkpoint_gradient_analysis.py') && kill $pids
  if procfind.py --quiet 'run_a53_reduced.sh'; then echo running; fi
"""
import os
import re
import sys

PROC = "/proc"


def cmdline(pid):
    try:
        with open(f"{PROC}/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (OSError, IOError):
        return None


def ppid_of(pid):
    try:
        with open(f"{PROC}/{pid}/stat", "rb") as f:
            data = f.read().decode("utf-8", "replace")
        # comm may contain spaces and parentheses; fields start after the last ')'
        return int(data[data.rindex(")") + 1:].split()[1])
    except (OSError, IOError, ValueError, IndexError):
        return 0


def ancestry(pid):
    """This process and every ancestor up to init, terminating on cycles."""
    chain, seen = [], set()
    while pid and pid not in seen:
        seen.add(pid)
        chain.append(pid)
        pid = ppid_of(pid)
    return chain


def main(argv):
    quiet = False
    if "--quiet" in argv:
        quiet = True
        argv = [a for a in argv if a != "--quiet"]
    mode = "substr"
    if argv and argv[0] in ("--re", "--exe"):
        mode, argv = argv[0][2:], argv[1:]
    if len(argv) != 1 or not argv[0]:
        sys.stderr.write(__doc__.split("USAGE")[1])
        return 4
    pat = argv[0]

    if mode == "re":
        try:
            rx = re.compile(pat)
        except re.error as e:
            sys.stderr.write(f"procfind: bad regex: {e}\n")
            return 4
        def hit(c):
            return bool(rx.search(c))
    elif mode == "exe":
        def hit(c):
            return bool(c) and os.path.basename(c.split(" ")[0]) == pat
    else:
        def hit(c):
            return pat in c

    mine = ancestry(os.getpid())
    # REFUSAL COMES FIRST. If the pattern describes the CALLER, no answer is
    # safe. Our own argv is skipped: it contains the pattern by construction,
    # since the pattern was passed to us on the command line. Every ancestor is
    # checked, and the invoking shell is the one that matters -- that is the
    # process `pgrep -f` handed back and `kill -9` destroyed.
    for pid in mine[1:]:
        c = cmdline(pid)
        if c and hit(c):
            if not quiet:
                sys.stderr.write(
                    f"procfind: REFUSED -- the pattern matches this process's own "
                    f"chain (pid {pid}):\n  {c[:200]}\n"
                    f"procfind: a pattern that names the caller cannot identify a "
                    f"target. Narrow it, or use --exe.\n")
            return 3

    exclude = set(mine)
    found = []
    for entry in os.listdir(PROC):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in exclude:
            continue
        c = cmdline(pid)
        if c and hit(c):
            found.append(pid)

    if not found:
        return 1
    if not quiet:
        print("\n".join(str(p) for p in sorted(found)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
