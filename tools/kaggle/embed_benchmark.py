import base64
import os
import re

with open("scripts/benchmark_model_vs_sam_tp.py", "rb") as f:
    bench_b64 = base64.b64encode(f.read()).decode("ascii")

with open("ai/traversability/dense_predictor.py", "rb") as f:
    dense_b64 = base64.b64encode(f.read()).decode("ascii")

with open("ai/traversability/sam_tp.py", "rb") as f:
    sam_b64 = base64.b64encode(f.read()).decode("ascii")

with open("tools/kaggle/ugv_gpu_runner.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace or inject EMBEDDED constants
for const_name, const_val in [
    ("EMBEDDED_BENCHMARK_B64", bench_b64),
    ("EMBEDDED_DENSE_PREDICTOR_B64", dense_b64),
    ("EMBEDDED_SAM_TP_B64", sam_b64),
]:
    if f"{const_name} = " in code:
        code = re.sub(rf'{const_name} = ".*?"', f'{const_name} = "{const_val}"', code, flags=re.DOTALL)
    else:
        anchor = 'REPO_NAME = "umangkaushik17-bit/Vision-Based-Autonomous-Navigation-for-UGV"'
        code = code.replace(anchor, f'{anchor}\n\n{const_name} = "{const_val}"')

new_setup = '''def setup_workspace():
    print("\\n[1/3] Setting up workspace codebase...")
    os.makedirs(REPO_DIR, exist_ok=True)

    print(f"Scanning {INPUT_ROOT} for attached datasets...")
    if os.path.exists(INPUT_ROOT):
        print(f"Attached inputs: {os.listdir(INPUT_ROOT)}")

        # Step 1: Copy any loose directory trees from sih2026-ugv-code
        for root, dirs, files in os.walk(INPUT_ROOT):
            if "ai" in dirs and any(k in files for k in ["setup.py", "pyproject.toml"]):
                print(f"Found project tree in {root}. Copying to {REPO_DIR} ...")
                shutil.copytree(root, REPO_DIR, dirs_exist_ok=True)
                break

        # Step 2: Unpack any attached zip archives (code.zip, checkpoints.zip, benchmark_test.zip)
        for root, dirs, files in os.walk(INPUT_ROOT):
            for f in files:
                f_lower = f.lower()
                zpath = os.path.join(root, f)
                if f_lower == "checkpoints.zip":
                    print(f"Found checkpoints archive: {zpath}. Extracting...")
                    for target_dir in [
                        os.path.join(REPO_DIR, "outputs", "rellis3d", "checkpoints"),
                        os.path.join(REPO_DIR, "rellis3d_gpu_ready", "checkpoints"),
                        os.path.join(REPO_DIR, "models", "rellis3d"),
                    ]:
                        os.makedirs(target_dir, exist_ok=True)
                        with zipfile.ZipFile(zpath, "r") as zf:
                            zf.extractall(target_dir)
                    print(f"[+] Checkpoints extracted to staging directories.")
                elif f_lower == "benchmark_test.zip":
                    print(f"Found benchmark_test archive: {zpath}. Extracting...")
                    target_bench = os.path.join(REPO_DIR, "data", "benchmark_test")
                    os.makedirs(target_bench, exist_ok=True)
                    with zipfile.ZipFile(zpath, "r") as zf:
                        zf.extractall(target_bench)
                    print(f"[+] Benchmark test data extracted to {target_bench}.")
                elif f_lower.endswith(".zip") and f_lower not in ("checkpoints.zip", "benchmark_test.zip"):
                    print(f"Found generic code archive: {zpath}. Extracting over {REPO_DIR} ...")
                    try:
                        with zipfile.ZipFile(zpath, "r") as zf:
                            zf.extractall(REPO_DIR)
                        print(f"[+] Successfully extracted {zpath}")
                    except Exception as e:
                        print(f"[-] Error extracting {zpath}: {e}")

    # Step 3: Restore embedded dense_predictor.py
    dense_dest = os.path.join(REPO_DIR, "ai", "traversability", "dense_predictor.py")
    os.makedirs(os.path.dirname(dense_dest), exist_ok=True)
    with open(dense_dest, "wb") as f:
        f.write(base64.b64decode(EMBEDDED_DENSE_PREDICTOR_B64))
    print(f"[+] Staged ai/traversability/dense_predictor.py ({os.path.getsize(dense_dest)} bytes)")

    # Step 4: Restore embedded sam_tp.py
    sam_dest = os.path.join(REPO_DIR, "ai", "traversability", "sam_tp.py")
    os.makedirs(os.path.dirname(sam_dest), exist_ok=True)
    with open(sam_dest, "wb") as f:
        f.write(base64.b64decode(EMBEDDED_SAM_TP_B64))
    print(f"[+] Staged ai/traversability/sam_tp.py ({os.path.getsize(sam_dest)} bytes)")

    # Step 5: Restore embedded benchmark_model_vs_sam_tp.py
    bench_dest = os.path.join(REPO_DIR, "scripts", "benchmark_model_vs_sam_tp.py")
    os.makedirs(os.path.dirname(bench_dest), exist_ok=True)
    with open(bench_dest, "wb") as f:
        f.write(base64.b64decode(EMBEDDED_BENCHMARK_B64))
    print(f"[+] Staged scripts/benchmark_model_vs_sam_tp.py ({os.path.getsize(bench_dest)} bytes)")

    # Step 6: Verify checkpoint best.pt
    found_ckpt = False
    for p in [
        os.path.join(REPO_DIR, "outputs", "rellis3d", "checkpoints", "best.pt"),
        os.path.join(REPO_DIR, "rellis3d_gpu_ready", "checkpoints", "best.pt"),
        os.path.join(REPO_DIR, "models", "rellis3d", "best.pt")
    ]:
        if os.path.exists(p):
            print(f"[+] Verified checkpoint at: {p} ({os.path.getsize(p)} bytes)")
            found_ckpt = True

    if not found_ckpt:
        for root, dirs, files in os.walk(INPUT_ROOT):
            for f in files:
                if f == "best.pt":
                    src_ckpt = os.path.join(root, f)
                    for dst_ckpt in [
                        os.path.join(REPO_DIR, "outputs", "rellis3d", "checkpoints", "best.pt"),
                        os.path.join(REPO_DIR, "rellis3d_gpu_ready", "checkpoints", "best.pt"),
                        os.path.join(REPO_DIR, "models", "rellis3d", "best.pt"),
                    ]:
                        os.makedirs(os.path.dirname(dst_ckpt), exist_ok=True)
                        shutil.copy2(src_ckpt, dst_ckpt)
                    print(f"[+] Found and staged checkpoint: {src_ckpt}")

    if os.path.exists(REPO_DIR):
        os.chdir(REPO_DIR)
        print(f"Current working directory: {os.getcwd()}")
        scripts_dir = os.path.join(REPO_DIR, "scripts")
        if os.path.exists(scripts_dir):
            print("Files in scripts/:", [f for f in os.listdir(scripts_dir) if "benchmark" in f or "rellis" in f][:10])

        pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f"{REPO_DIR}:{pythonpath}"
        sys.path.insert(0, REPO_DIR)'''

# Replace setup_workspace implementation
pattern = r"def setup_workspace\(\):.*?(?=def test_dynamic_gpu)"
code = re.sub(pattern, new_setup + "\n\n\n", code, flags=re.DOTALL)

with open("tools/kaggle/ugv_gpu_runner.py", "w", encoding="utf-8") as f:
    f.write(code)

print("ugv_gpu_runner.py updated with all embedded components!")
