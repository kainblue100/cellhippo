"""Download a large file over many ranged connections.

The 10x CloudFront throttles a single connection to ~0.2 MB/s, but the server's
own link is 10 MB/s and four parallel ranges already measured 1.48 MB/s.  So the
bottleneck is per-connection throttling, not bandwidth, and the fix is to open
many ranged downloads and stitch them.

Each part is written to its own file and retried independently, so an interrupted
run resumes rather than starting over.  The parts are concatenated only when every
one of them has the expected length.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def remote_size(url: str) -> int:
    r = subprocess.run(["curl", "-sIL", "-m", "30", url],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return 0


def fetch(url: str, out: str, start: int, end: int) -> bool:
    tmp = f"{out}.tmp"
    cmd = ["curl", "-sL", "--retry", "4", "--retry-delay", "3",
           "-r", f"{start}-{end}", "-o", tmp, url]
    subprocess.run(cmd, timeout=3600)
    want = end - start + 1
    got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if got == want:
        os.replace(tmp, out)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("-n", "--parts", type=int, default=16)
    a = ap.parse_args()

    total = remote_size(a.url)
    if not total:
        sys.exit("could not read content-length")
    print(f"remote size {total/1e6:.1f} MB, {a.parts} parts", flush=True)

    size = total // a.parts
    ranges = []
    for i in range(a.parts):
        s = i * size
        e = (total - 1) if i == a.parts - 1 else (s + size - 1)
        ranges.append((f"{a.out}.part{i:02d}", s, e))

    # one round of parallel fetches; then retry whatever is still missing
    for attempt in range(1, 9):
        todo = [(o, s, e) for o, s, e in ranges
                if not (os.path.exists(o) and os.path.getsize(o) == e - s + 1)]
        if not todo:
            break
        print(f"  attempt {attempt}: {len(todo)} part(s) outstanding", flush=True)
        procs = []
        for o, s, e in todo:
            procs.append((o, s, e, subprocess.Popen(
                ["curl", "-sL", "--retry", "3", "-r", f"{s}-{e}",
                 "-o", f"{o}.tmp", a.url])))
        t0 = time.time()
        for o, s, e, p in procs:
            p.wait()
            want = e - s + 1
            got = os.path.getsize(f"{o}.tmp") if os.path.exists(f"{o}.tmp") else 0
            if got == want:
                os.replace(f"{o}.tmp", o)
        got = sum(os.path.getsize(o) for o, _, _ in ranges
                  if os.path.exists(o))
        el = time.time() - t0
        print(f"           {got/1e6:.0f}/{total/1e6:.0f} MB "
              f"({got/max(el,1e-6)/1e6:.2f} MB/s this round)", flush=True)

    missing = [o for o, s, e in ranges
               if not (os.path.exists(o) and os.path.getsize(o) == e - s + 1)]
    if missing:
        sys.exit(f"incomplete parts: {missing}")

    with open(a.out, "wb") as fh:
        for o, _, _ in ranges:
            with open(o, "rb") as p:
                while True:
                    chunk = p.read(1 << 24)
                    if not chunk:
                        break
                    fh.write(chunk)
    print(f"wrote {a.out}  {os.path.getsize(a.out)/1e6:.1f} MB")
    for o, _, _ in ranges:
        os.remove(o)


if __name__ == "__main__":
    main()
