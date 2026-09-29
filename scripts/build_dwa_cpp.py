"""Build script for Native C++ DWA Local Planner Backend.

Automatically detects MSVC on Windows (Visual Studio / BuildTools) or GCC/Clang on Linux,
and compiles dwa_core.cpp + dwa_c_api.cpp into an optimized shared library with AVX2 and OpenMP.

Usage:
    python scripts/build_dwa_cpp.py
"""

import os
import platform
import subprocess
import sys


def find_vcvarsall() -> str:
    """Locate vcvarsall.bat on Windows using vswhere or standard installation paths."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    # Try vswhere
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        try:
            out = subprocess.check_output([
                vswhere, "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"
            ], text=True).strip()
            if out:
                vcvars = os.path.join(out, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                if os.path.exists(vcvars):
                    return vcvars
        except Exception:
            pass

    return ""


def build_cpp_backend():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cpp_dir = os.path.join(repo_root, "ai", "planning", "native_cpp")

    src_core = os.path.join(cpp_dir, "dwa_core.cpp")
    src_api = os.path.join(cpp_dir, "dwa_c_api.cpp")

    if not os.path.exists(src_core) or not os.path.exists(src_api):
        print(f"[ERROR] Source files missing in {cpp_dir}")
        sys.exit(1)

    system = platform.system()
    print("=" * 70)
    print(f"BUILDING NATIVE C++ DWA CPU BACKEND ON {system.upper()}")
    print("=" * 70)

    if system == "Windows":
        target_dll = os.path.join(cpp_dir, "dwa_cpp.dll")
        vcvarsall = find_vcvarsall()
        if not vcvarsall:
            print("[ERROR] MSVC vcvarsall.bat could not be found.")
            sys.exit(1)

        print(f"[*] Found MSVC Environment Tool: {vcvarsall}")
        # Build command:
        # /O2 : Max speed
        # /std:c++17 : Modern C++17
        # /openmp : Multithreaded candidate evaluation
        # /arch:AVX2 : AVX2 vector instructions
        # /fp:fast : Fast floating-point math
        # /EHsc : C++ exception handling
        # /LD : Create DLL
        cmd = f'call "{vcvarsall}" x64 && cl.exe /O2 /std:c++17 /openmp /arch:AVX2 /fp:fast /EHsc /LD /Fe:"{target_dll}" "{src_core}" "{src_api}"'
        print(f"[*] Executing MSVC compilation...")
        res = subprocess.run(cmd, shell=True, cwd=cpp_dir)
        if res.returncode != 0:
            print("[ERROR] MSVC compilation failed!")
            sys.exit(res.returncode)

        # Cleanup intermediate .obj and .exp files
        for ext in [".obj", ".exp", ".lib"]:
            for f in os.listdir(cpp_dir):
                if f.endswith(ext):
                    try:
                        os.remove(os.path.join(cpp_dir, f))
                    except Exception:
                        pass

        if os.path.exists(target_dll):
            size_kb = os.path.getsize(target_dll) / 1024.0
            print(f"[SUCCESS] Built {target_dll} ({size_kb:.1f} KB)")
        else:
            print("[ERROR] Target DLL was not created!")
            sys.exit(1)

    else:  # Linux / Unix
        target_so = os.path.join(cpp_dir, "libdwa_cpp.so")
        cmd = [
            "g++", "-O3", "-std=c++17", "-fopenmp", "-mavx2", "-ffast-math",
            "-fPIC", "-shared", "-o", target_so, src_core, src_api
        ]
        print(f"[*] Executing GCC compilation: {' '.join(cmd)}")
        res = subprocess.run(cmd, cwd=cpp_dir)
        if res.returncode != 0:
            print("[ERROR] GCC compilation failed!")
            sys.exit(res.returncode)

        if os.path.exists(target_so):
            size_kb = os.path.getsize(target_so) / 1024.0
            print(f"[SUCCESS] Built {target_so} ({size_kb:.1f} KB)")
        else:
            print("[ERROR] Target shared library was not created!")
            sys.exit(1)

    print("=" * 70)
    print("NATIVE C++ DWA BUILD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    build_cpp_backend()
