from pathlib import Path
import importlib
import inspect
import os
import runpy
import sys
import torch

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))

module = importlib.import_module("pyro.distributions.transforms.spline")
original = module._monotonic_rational_spline
signature = inspect.signature(original)
checked = False

def safe_tail_spline(*args, **kwargs):
    global checked
    call = signature.bind(*args, **kwargs)
    call.apply_defaults()

    inputs = call.arguments["inputs"]
    bound = call.arguments["bound"]
    inside = (inputs >= -bound) & (inputs <= bound)

    call.arguments["inputs"] = torch.where(
        inside, inputs, torch.zeros_like(inputs)
    )
    outputs, logdet = original(*call.args, **call.kwargs)
    outputs = torch.where(inside, outputs, inputs)
    logdet = torch.where(inside, logdet, torch.zeros_like(logdet))

    if call.arguments["inverse"] and not checked:
        with torch.no_grad():
            old_outputs, old_logdet = original(*args, **kwargs)
            torch.testing.assert_close(
                outputs.detach(), old_outputs, rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(
                logdet.detach(), old_logdet, rtol=1e-5, atol=1e-6
            )
        print(
            "[CHECK] inverse outputs/logdet match; outside count=",
            (~inside).sum().item(),
            flush=True,
        )
        checked = True

    return outputs, logdet

module._monotonic_rational_spline = safe_tail_spline
print("[PATCH] safe spline tail handling enabled", flush=True)

if os.environ.get("CHECK_RESUME_STEP"):
    source = sys.argv[sys.argv.index("--resume") + 1]
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    expected = int(os.environ["CHECK_RESUME_STEP"])
    actual = int(ckpt["step"])
    print(f"[CHECK] checkpoint step={actual}", flush=True)
    if actual != expected:
        raise SystemExit(f"Expected step {expected}, found {actual}")
    del ckpt

sys.argv[0] = str(root / "cloud_generator_step.py")
if os.environ.get("ANOMALY_DEBUG") == "1":
    with torch.autograd.detect_anomaly(check_nan=True):
        runpy.run_path(sys.argv[0], run_name="__main__")
else:
    runpy.run_path(sys.argv[0], run_name="__main__")
