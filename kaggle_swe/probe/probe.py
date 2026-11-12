"""CPU probe: install the current wheelhouse's harness packages (swegemma, adk_submission, adk_eval_core) into the output
folder, so their source can be read locally, and print the interfaces the compare stage uses."""
import glob
import importlib
import inspect
import subprocess
import sys

wheels = [w for w in glob.glob("/kaggle/input/**/gemma-4-developer-agent-wheelhouse/*.whl", recursive=True)
          if any(k in w.split("/")[-1] for k in ("swegemma", "adk_submission", "adk_eval_core"))]
print("wheels:", [w.split("/")[-1] for w in wheels], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", "/kaggle/working/pkgs", *wheels], check=True)
sys.path.insert(0, "/kaggle/working/pkgs")
for mod, names in (("swegemma.models", None), ("swegemma.config", ["EvalConfig", "build_submission_limits"]),
                   ("swegemma.evaluate", ["Evaluator"])):
    try:
        m = importlib.import_module(mod)
        print(f"== {mod}: exports", sorted(n for n in dir(m) if not n.startswith("_"))[:60], flush=True)
        for n in names or []:
            obj = getattr(m, n, None)
            print(f"   {n}:", inspect.signature(obj) if obj is not None and callable(obj) else obj, flush=True)
    except Exception as error:  # print and go on: the point is to see what is there
        print(f"== {mod}: import failed: {error!r}", flush=True)
print(subprocess.run(["grep", "-rn", "def setup_gemma_model_registry\|def validate_single_declared_model\|def create_model_registry",
                      "/kaggle/working/pkgs"], capture_output=True, text=True).stdout, flush=True)
