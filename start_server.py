import subprocess
import time
import sys
import os
import urllib.request

try:
    urllib.request.urlopen("http://localhost:7860/statue", timeout=1)
    print("Server is already running!")
    sys.exit(0)
except Exception:
    pass

env = os.environ.copy()
env["PYTHONNOUSERSITE"] = "1"
env["PYOPENGL_PLATFORM"] = "egl"

log = open("/home/braitoli/workspace/namnh/code/poc/UniRig/playground/server.log", "a")
p = subprocess.Popen(
    ["/home/braitoli/miniconda/envs/unirig312/bin/python", "playground/server.py"],
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd="/home/braitoli/workspace/namnh/code/poc/UniRig",
    start_new_session=True,
    env=env
)
print(f"Detached background server started with PID: {p.pid}")
for _ in range(15):
    time.sleep(1)
    try:
        urllib.request.urlopen("http://localhost:7860/statue", timeout=1)
        print("Server is verified healthy and responding on http://localhost:7860/statue!")
        sys.exit(0)
    except Exception:
        pass
print("Server took longer than expected to become ready.")
