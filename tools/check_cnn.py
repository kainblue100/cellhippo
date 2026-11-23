"""Can we get a real CNN image encoder, or do we fall back to handcrafted colour
statistics?

HIPPO encodes 224x224 H&E tiles with a ResNet50.  If torch + torchvision weights
are reachable we can match that; the server reaches PyPI but not HF or GitHub, so
download.pytorch.org is the question.
"""
import subprocess
import sys

V = "/mnt/storage/bzhang/AIVC/vcellx/.venv/bin/python"


def run(code, timeout=600):
    p = subprocess.run([V, "-c", code], capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


print("=== torch ===")
rc, out, err = run("import torch, torchvision; "
                   "print('torch', torch.__version__); "
                   "print('torchvision', torchvision.__version__); "
                   "print('cuda', torch.cuda.is_available())")
print(out or err)

print("\n=== can we download pretrained ResNet50 weights? ===")
rc, out, err = run("""
import torch, torchvision, time
t0 = time.time()
try:
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    n = sum(p.numel() for p in m.parameters())
    print(f'OK resnet50 loaded, {n/1e6:.1f}M params, {time.time()-t0:.1f}s')
except Exception as e:
    print('FAIL', type(e).__name__, str(e)[:200])
""", timeout=900)
print(out or err)

print("\n=== host reachability ===")
rc, out, err = run("""
import urllib.request
for u in ('https://download.pytorch.org',
          'https://pypi.org',
          'https://huggingface.co'):
    try:
        r = urllib.request.urlopen(u, timeout=20)
        print(f'  {u:36s} OK {r.status}')
    except Exception as e:
        print(f'  {u:36s} FAIL {type(e).__name__}')
""")
print(out or err)
