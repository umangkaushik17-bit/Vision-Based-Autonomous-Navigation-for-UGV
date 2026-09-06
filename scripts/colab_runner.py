"""Google Colab Bridge & Remote Execution Manager for Antigravity IDE.

Enables local AI agent and developer to:
1. Update SSH tunnel configuration (Cloudflare tunnel host)
2. Test connection and inspect remote GPU
3. Synchronize local workspace to Colab
4. Execute remote commands on Colab GPU
5. Pull generated artifacts/metrics/plots back to local workspace
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

SSH_CONFIG_PATH = os.path.expanduser("~/.ssh/config")
CLOUDFLARED_PATH = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"


def update_colab_host(hostname_or_url: str) -> bool:
    """Extract and update the HostName in ~/.ssh/config."""
    # Clean input: extract *.trycloudflare.com or ip
    match = re.search(r"([a-zA-Z0-9-]+\.trycloudflare\.com)", hostname_or_url)
    if match:
        hostname = match.group(1)
    else:
        hostname = hostname_or_url.strip()

    if not hostname:
        print("[-] Error: Invalid HostName provided.")
        return False

    config_content = f"""# Google Colab SSH Bridge for Antigravity IDE
Host colab
    HostName {hostname}
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ProxyCommand "{CLOUDFLARED_PATH}" access ssh --hostname %h
    ServerAliveInterval 30
"""
    try:
        with open(SSH_CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"[+] Successfully updated ~/.ssh/config:")
        print(f"    Host colab -> {hostname}")
        return True
    except Exception as e:
        print(f"[-] Failed to write {SSH_CONFIG_PATH}: {e}")
        return False


def test_connection() -> bool:
    """Verify SSH connectivity and query GPU state."""
    print("[*] Testing connection to Colab (ssh colab)...")
    cmd = ["ssh", "colab", "nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if proc.returncode == 0:
            gpu_info = proc.stdout.strip()
            print("[+] Connection SUCCESSFUL! Remote GPU detected:")
            print(f"    {gpu_info}")
            return True
        else:
            print("[-] Connection FAILED:")
            print(proc.stderr.strip() or proc.stdout.strip())
            return False
    except subprocess.TimeoutExpired:
        print("[-] Connection timed out (check if Colab tunnel is running and HostName is correct).")
        return False
    except Exception as e:
        print(f"[-] Error: {e}")
        return False


def run_command(command: str) -> int:
    """Execute a shell command inside /content/SIH2026 on Colab."""
    print(f"[*] Executing on Colab: {command}")
    remote_cmd = f"cd /content/SIH2026 && {command}"
    proc = subprocess.run(["ssh", "colab", remote_cmd])
    return proc.returncode


def sync_local_to_colab() -> int:
    """Copy current local workspace code to /content/SIH2026 on Colab."""
    print("[*] Ensuring /content/SIH2026 exists on Colab...")
    subprocess.run(["ssh", "colab", "mkdir -p /content/SIH2026"])

    print("[*] Synchronizing code and directories to Colab...")
    # List of directories to sync
    sync_dirs = ["ai", "simulation", "scripts", "validation", "tests", "docs"]
    sync_files = ["pyproject.toml", "README.md"]

    for d in sync_dirs:
        if os.path.exists(d):
            print(f"    -> Syncing {d}/ ...")
            subprocess.run(["scp", "-r", d, f"colab:/content/SIH2026/{d}"])

    for f in sync_files:
        if os.path.exists(f):
            print(f"    -> Syncing {f} ...")
            subprocess.run(["scp", f, f"colab:/content/SIH2026/{f}"])

    print("[+] Synchronization complete.")
    return 0


def pull_from_colab(remote_path: str = "simulation/outputs/", local_path: str = "simulation/outputs/") -> int:
    """Pull artifacts from Colab back to local machine."""
    print(f"[*] Pulling colab:/content/SIH2026/{remote_path} -> {local_path} ...")
    os.makedirs(local_path, exist_ok=True)
    proc = subprocess.run(["scp", "-r", f"colab:/content/SIH2026/{remote_path}", local_path])
    if proc.returncode == 0:
        print("[+] Artifacts successfully pulled.")
    else:
        print(f"[-] Pull failed with code {proc.returncode}")
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Antigravity Google Colab Bridge")
    parser.add_argument("--set-host", type=str, help="Set new Colab HostName (e.g. xxx.trycloudflare.com)")
    parser.add_argument("--status", action="store_true", help="Test connection and query remote GPU")
    parser.add_argument("--sync", action="store_true", help="Sync local code to Colab /content/SIH2026")
    parser.add_argument("--run", type=str, help="Run command on Colab in /content/SIH2026")
    parser.add_argument("--pull", type=str, nargs="?", const="simulation/outputs/", help="Pull results back to local")

    args = parser.parse_args()

    if args.set_host:
        update_colab_host(args.set_host)
        test_connection()
    elif args.status:
        test_connection()
    elif args.sync:
        sync_local_to_colab()
    elif args.run:
        run_command(args.run)
    elif args.pull:
        pull_from_colab(remote_path=args.pull)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
