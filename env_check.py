
"""VisionPort Session 4 - Environment Check"""

import platform
import shutil
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

rows = []


def log(key, value):
    rows.append((key, str(value)))
    print(f"{key:<30} {value}")


def pkg(name):
    try:
        return version(name)
    except Exception:
        return "NOT INSTALLED"


# 1. System and Python
print("\n========== SYSTEM AND PYTHON ==========\n")

log("Date", datetime.now().strftime("%Y-%m-%d %H:%M"))
log("Operating system", platform.platform())
log("Python version", sys.version.split()[0])
log("Python interpreter", sys.executable)


# 2. Package versions
print("\n========== PACKAGE VERSIONS ==========\n")

packages = [
    "torch",
    "torchvision",
    "ultralytics",
    "numpy",
    "pillow",
    "fastapi",
    "uvicorn",
    "streamlit",
    "pandas"
]

for name in packages:
    log(f"pkg: {name}", pkg(name))


# 3. NVIDIA driver
print("\n========== NVIDIA DRIVER ==========\n")

if shutil.which("nvidia-smi"):
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if out.returncode == 0 and out.stdout.strip():
            log("NVIDIA driver", out.stdout.strip().splitlines()[0])
        else:
            log("NVIDIA driver", "Unable to read driver information")

    except Exception as e:
        log("NVIDIA driver", f"ERROR: {type(e).__name__}")
else:
    log("NVIDIA driver", "nvidia-smi not found (no NVIDIA GPU or driver)")


# 4. PyTorch, CUDA, cuDNN and GPU
print("\n========== PYTORCH AND GPU ==========\n")

try:
    import torch

    log("PyTorch CUDA build", torch.version.cuda or "None (CPU-only build)")
    log("torch.cuda.is_available()", torch.cuda.is_available())

    if torch.backends.cudnn.is_available():
        log("cuDNN version", torch.backends.cudnn.version())
    else:
        log("cuDNN version", "not available")

    if torch.cuda.is_available():

        log("GPU count", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)

            log(
                f"GPU {i}",
                f"{p.name}, {p.total_memory / 1e9:.1f} GB, "
                f"compute capability {p.major}.{p.minor}"
            )

        # GPU matrix multiplication test
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x

        torch.cuda.synchronize()

        log("GPU matmul test", f"PASS (result on {y.device})")

    else:
        log("GPU matmul test", "SKIPPED (running on CPU)")

except Exception as e:
    log("PyTorch", f"FAILED to import or run: {type(e).__name__}")


# 5. Save environment report
print("\n========== SAVING REPORT ==========\n")

lines = [
    "# VisionPort Environment Report",
    "",
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "",
    "| Item | Value |",
    "|---|---|"
]

lines += [f"| {key} | {value} |" for key, value in rows]

Path("env_report.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

print("Environment report saved successfully!")
print("File created: env_report.md")
