"""Kaggle GPU Cloud Automation Manager for Antigravity IDE.

Provides automated dispatching, monitoring, and output retrieval from Kaggle GPUs:
- Packages and syncs local workspace code directly into private Kaggle dataset
- Submits headless GPU workloads (NVIDIA Tesla P100 / T4)
- Monitors execution status (queued, running, complete, error)
- Retrieves logs and output artifacts (metrics JSON, plots) into simulation/outputs/kaggle/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile

KAGGLE_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_JSON = os.path.join(KAGGLE_DIR, "kaggle.json")
TOOLS_DIR = os.path.join("tools", "kaggle")
METADATA_PATH = os.path.join(TOOLS_DIR, "kernel-metadata.json")
RUNNER_PATH = os.path.join(TOOLS_DIR, "ugv_gpu_runner.py")
DATASET_DIR = os.path.join(TOOLS_DIR, "dataset")
DATASET_META = os.path.join(DATASET_DIR, "dataset-metadata.json")
OUTPUT_DIR = os.path.join("simulation", "outputs", "kaggle")


def init_credentials() -> tuple[str, str]:
    """Load credentials from ~/.kaggle/kaggle.json and configure environment."""
    if not os.path.exists(KAGGLE_JSON):
        print(f"[-] Error: {KAGGLE_JSON} not found.")
        print("    Please run: python scripts/kaggle_runner.py --setup-token '<token>' --username '<username>'")
        sys.exit(1)

    try:
        with open(KAGGLE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        username = str(data.get("username", "")).strip()
        key = str(data.get("key", "")).strip()

        os.environ["KAGGLE_API_TOKEN"] = key
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key

        return username, key
    except Exception as e:
        print(f"[-] Error parsing {KAGGLE_JSON}: {e}")
        sys.exit(1)


def get_kaggle_api():
    """Authenticate and return the Kaggle API instance."""
    init_credentials()
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def package_code() -> str:
    """Create a lean zip package of local code (ai, simulation, tests, validation, configs)."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, "code.zip")
    include_dirs = ["ai", "simulation", "tests", "validation", "config", "scripts", "docs", "ros2_ws"]

    print(f"[*] Packaging local code into {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in include_dirs:
            if not os.path.exists(d):
                continue
            for root, _, files in os.walk(d):
                if any(p in root for p in ["__pycache__", "outputs", ".pytest_cache", ".venv"]):
                    continue
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    fp = os.path.join(root, f)
                    zf.write(fp, fp)
        for root_file in os.listdir("."):
            if os.path.isfile(root_file) and (
                root_file.endswith(".py")
                or root_file.endswith(".toml")
                or root_file.startswith("requirements")
                or root_file == "setup.py"
            ):
                zf.write(root_file, root_file)

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"[+] Code package created: {size_mb:.2f} MB")
    return zip_path


def sync_code(notes: str = "Sync local code changes") -> bool:
    """Package and upload code updates to Kaggle dataset."""
    username, _ = init_credentials()
    package_code()

    # Ensure dataset metadata
    dataset_id = f"{username}/sih2026-ugv-code"
    meta = {
        "title": "SIH2026 UGV Codebase",
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(DATASET_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    api = get_kaggle_api()
    print(f"[*] Updating Kaggle dataset {dataset_id} ...")
    # Temporarily cd into DATASET_DIR to avoid Windows pathing bug in Kaggle upload
    cur_dir = os.getcwd()
    try:
        os.chdir(DATASET_DIR)
        api.dataset_create_version(".", version_notes=notes, quiet=False)
        print("[+] Code dataset uploaded. Waiting for Kaggle processing...")
        for _ in range(30):
            time.sleep(3)
            st = str(api.dataset_status(dataset_id)).lower()
            if "ready" in st:
                print(f"[+] Dataset {dataset_id} is ready for kernel mounting!")
                break
            print(f"    Dataset status: {st} ...")
        return True
    except Exception as e:
        print(f"[-] Dataset version update note: {e}")
        return False
    finally:
        os.chdir(cur_dir)


def sync_kernel_metadata(username: str, kernel_slug: str = "sih2026-ugv-gpu-runner") -> str:
    """Update kernel-metadata.json with active username, slug, and dataset source."""
    kernel_id = f"{username}/{kernel_slug}"
    meta = {
        "id": kernel_id,
        "title": "SIH2026 UGV GPU Runner",
        "code_file": "ugv_gpu_runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [
            f"{username}/sih2026-ugv-code"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    os.makedirs(TOOLS_DIR, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return kernel_id


def setup_token(username: str, token_key: str) -> bool:
    """Configure ~/.kaggle/kaggle.json with username and key."""
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    data = {"username": username.strip(), "key": token_key.strip()}
    try:
        with open(KAGGLE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[+] Successfully saved Kaggle API token to {KAGGLE_JSON} (User: {username})")
        sync_kernel_metadata(username)
        return True
    except Exception as e:
        print(f"[-] Error writing token: {e}")
        return False


def submit_kernel(custom_cmd: str | None = None) -> bool:
    """Push kernel to Kaggle cloud GPU."""
    username, _ = init_credentials()
    kernel_id = sync_kernel_metadata(username)

    if custom_cmd:
        with open(RUNNER_PATH, "r", encoding="utf-8") as f:
            code = f.read()
        escaped_cmd = json.dumps(custom_cmd)
        code = re.sub(
            r'DEFAULT_CMD\s*=\s*.*',
            f'DEFAULT_CMD = {escaped_cmd}',
            code,
        )
        with open(RUNNER_PATH, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[*] Configured execution command: {custom_cmd}")

    print(f"[*] Submitting kernel {kernel_id} to Kaggle Cloud GPU...")
    api = get_kaggle_api()
    try:
        res = api.kernels_push(TOOLS_DIR)
        url = getattr(res, "url", f"/code/{kernel_id}")
        version = getattr(res, "version_number", getattr(res, "versionNumber", 1))
        print(f"[+] Kernel submitted successfully!")
        print(f"    URL: https://www.kaggle.com{url}")
        print(f"    Version: {version}")
        return True
    except Exception as e:
        print(f"[-] Submission failed: {e}")
        return False


def check_status() -> dict:
    """Query current execution status of the Kaggle kernel."""
    username, _ = init_credentials()
    kernel_id = f"{username}/sih2026-ugv-gpu-runner"
    api = get_kaggle_api()
    try:
        status_info = api.kernels_status(kernel_id)
        msg = ""

        if hasattr(status_info, "status"):
            st = status_info.status
            status = getattr(st, "name", str(st).split(".")[-1]).upper()
            msg = getattr(status_info, "failure_message", getattr(status_info, "failureMessage", "")) or ""
        elif isinstance(status_info, dict):
            status = str(status_info.get("status", "UNKNOWN")).upper()
            msg = str(status_info.get("failureMessage", ""))
        else:
            status = str(status_info).upper()

        print(f"[*] Status: {status}" + (f" (Message: {msg})" if msg else ""))
        return {"status": status, "failureMessage": str(msg)}
    except Exception as e:
        print(f"[-] Status check transient network note: {e}")
        return {"status": "TRANSIENT_NET_ERROR", "failureMessage": str(e)}


def download_outputs() -> bool:
    """Download generated output artifacts from Kaggle to local simulation/outputs/kaggle/."""
    username, _ = init_credentials()
    kernel_id = f"{username}/sih2026-ugv-gpu-runner"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[*] Downloading outputs from {kernel_id} to {OUTPUT_DIR} ...")
    api = get_kaggle_api()
    for attempt in range(1, 4):
        try:
            res = api.kernels_output(kernel_id, OUTPUT_DIR)
            print(f"[+] Output download finished. Artifacts saved in {OUTPUT_DIR}")
            return True
        except Exception as e:
            print(f"[-] Download attempt {attempt}/3 failed: {e}")
            time.sleep(5)
    return False


def watch_running(poll_interval_s: int = 15) -> bool:
    """Monitor an already submitted Kaggle kernel until completion and download outputs."""
    print("\n[*] Monitoring execution of active Kaggle Cloud GPU kernel...")
    consecutive_net_errors = 0
    while True:
        status_dict = check_status()
        status = status_dict.get("status", "UNKNOWN")

        if status == "COMPLETE":
            print("\n[+] Kernel execution completed successfully!")
            download_outputs()
            return True
        elif status == "TRANSIENT_NET_ERROR":
            consecutive_net_errors += 1
            if consecutive_net_errors >= 10:
                print("\n[-] 10 consecutive network errors to Kaggle API. Aborting monitor.")
                return False
            print(f"    (Transient network error {consecutive_net_errors}/10. Retrying in {poll_interval_s}s...)")
            time.sleep(poll_interval_s)
        elif status in ("ERROR", "FAILED", "CANCELLED"):
            print(f"\n[-] Kernel execution terminated with status: {status}")
            if status_dict.get("failureMessage"):
                print(f"    Failure message: {status_dict.get('failureMessage')}")
            download_outputs()
            return False
        else:
            consecutive_net_errors = 0
            time.sleep(poll_interval_s)


def run_and_watch(custom_cmd: str | None = None, poll_interval_s: int = 15) -> bool:
    """Full lifecycle: Sync code -> Submit -> Poll until complete -> Download artifacts."""
    print("[*] Synchronizing local code with Kaggle cloud...")
    if not sync_code():
        return False
    if not submit_kernel(custom_cmd):
        return False

    return watch_running(poll_interval_s=poll_interval_s)


def main():
    parser = argparse.ArgumentParser(description="Antigravity Kaggle GPU Cloud Runner")
    parser.add_argument("--setup-token", type=str, help="Kaggle API key (e.g. KGAT_...)")
    parser.add_argument("--username", type=str, default="umangkaushik17", help="Kaggle username")
    parser.add_argument("--sync-code", action="store_true", help="Package and update code dataset on Kaggle")
    parser.add_argument("--submit", action="store_true", help="Submit kernel to Kaggle GPU")
    parser.add_argument("--status", action="store_true", help="Check execution status")
    parser.add_argument("--download", action="store_true", help="Download output artifacts")
    parser.add_argument("--run", type=str, help="Execute full lifecycle with custom command")

    parser.add_argument("--watch", action="store_true", help="Monitor currently running kernel until completion")

    args = parser.parse_args()

    if args.setup_token:
        setup_token(args.username, args.setup_token)
    elif args.sync_code:
        if not sync_code():
            sys.exit(1)
    elif args.submit:
        if not submit_kernel():
            sys.exit(1)
    elif args.status:
        check_status()
    elif args.watch:
        success = watch_running()
        if not success:
            sys.exit(1)
    elif args.download:
        if not download_outputs():
            sys.exit(1)
    elif args.run:
        success = run_and_watch(args.run)
        if not success:
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
