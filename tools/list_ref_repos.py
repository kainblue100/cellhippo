"""List the two reference repos so their image pipeline can be followed rather
than reinvented."""
import json
import subprocess

GH = r"D:\coda_lungs\tools\gh_extract\bin\gh.exe"


def api(path):
    p = subprocess.run([GH, "api", path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(p.stderr[:300])
    return json.loads(p.stdout)


for repo in ("ZhoulabCPH/HIPOS", "SydneyBioX/GHIST_plus"):
    print(f"\n{'='*70}\n{repo}\n{'='*70}")
    try:
        info = api(f"repos/{repo}")
        print(f"  branch {info['default_branch']}  private={info['private']}")
        t = api(f"repos/{repo}/git/trees/{info['default_branch']}?recursive=1")
    except RuntimeError as e:
        print("  ", e)
        continue
    blobs = [e for e in t["tree"] if e["type"] == "blob"]
    print(f"  {len(blobs)} files, {sum(e.get('size') or 0 for e in blobs)/1e6:.1f} MB")
    for e in sorted(blobs, key=lambda x: -(x.get("size") or 0)):
        p = e["path"]
        if any(p.lower().endswith(x) for x in
               (".py", ".md", ".yaml", ".yml", ".json", ".txt", ".ipynb")):
            print(f"    {(e.get('size') or 0)/1024:9.1f} KB  {p}")
