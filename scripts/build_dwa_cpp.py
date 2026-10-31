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
        vcvarsall = find_vcvarsall()
        if not vcvarsall:
            print("[ERROR] MSVC vcvarsall.bat could not be found.")
            sys.exit(1)

        print(f"[*] Found MSVC Environment Tool: {vcvarsall}")

        # 1. Build SIMD target (AVX2 + OpenMP)
        target_simd = os.path.join(cpp_dir, "dwa_cpp.dll")
        cmd_simd = f'call "{vcvarsall}" x64 && cl.exe /O2 /std:c++17 /openmp /arch:AVX2 /fp:fast /EHsc /LD /Fe:"{target_simd}" "{src_core}" "{src_api}"'
        print(f"[*] Compiling C++ SIMD (AVX2) target: {target_simd}")
        res_simd = subprocess.run(cmd_simd, shell=True, cwd=cpp_dir)
        if res_simd.returncode != 0:
            print("[ERROR] MSVC SIMD compilation failed!")
            sys.exit(res_simd.returncode)

        # 2. Build Scalar target (Portable standard C++, no AVX vector flags)
        target_scalar = os.path.join(cpp_dir, "dwa_cpp_scalar.dll")
        cmd_scalar = f'call "{vcvarsall}" x64 && cl.exe /O2 /std:c++17 /fp:fast /EHsc /LD /Fe:"{target_scalar}" "{src_core}" "{src_api}"'
        print(f"[*] Compiling C++ Scalar (Portable) target: {target_scalar}")
        res_scalar = subprocess.run(cmd_scalar, shell=True, cwd=cpp_dir)
        if res_scalar.returncode != 0:
            print("[ERROR] MSVC Scalar compilation failed!")
            sys.exit(res_scalar.returncode)

        # Cleanup intermediate .obj and .exp files
        for ext in [".obj", ".exp", ".lib"]:
            for f in os.listdir(cpp_dir):
                if f.endswith(ext):
                    try:
                        os.remove(os.path.join(cpp_dir, f))
                    except Exception:
                        pass

        for target in [target_simd, target_scalar]:
            if os.path.exists(target):
                size_kb = os.path.getsize(target) / 1024.0
                print(f"[SUCCESS] Built {os.path.basename(target)} ({size_kb:.1f} KB)")
            else:
                print(f"[ERROR] Target {target} was not created!")
                sys.exit(1)

    else:  # Linux / Unix
        # 1. Build SIMD target (AVX2 + OpenMP)
        target_simd = os.path.join(cpp_dir, "libdwa_cpp.so")
        cmd_simd = [
            "g++", "-O3", "-std=c++17", "-fopenmp", "-mavx2", "-ffast-math",
            "-fPIC", "-shared", "-o", target_simd, src_core, src_api
        ]
        print(f"[*] Compiling C++ SIMD (AVX2) target: {' '.join(cmd_simd)}")
        res_simd = subprocess.run(cmd_simd, cwd=cpp_dir)
        if res_simd.returncode != 0:
            print("[ERROR] GCC SIMD compilation failed!")
            sys.exit(res_simd.returncode)

        # 2. Build Scalar target (Portable standard C++, no -mavx2)
        target_scalar = os.path.join(cpp_dir, "libdwa_cpp_scalar.so")
        cmd_scalar = [
            "g++", "-O3", "-std=c++17", "-ffast-math",
            "-fPIC", "-shared", "-o", target_scalar, src_core, src_api
        ]
        print(f"[*] Compiling C++ Scalar target: {' '.join(cmd_scalar)}")
        res_scalar = subprocess.run(cmd_scalar, cwd=cpp_dir)
        if res_scalar.returncode != 0:
            print("[ERROR] GCC Scalar compilation failed!")
            sys.exit(res_scalar.returncode)

        for target in [target_simd, target_scalar]:
            if os.path.exists(target):
                size_kb = os.path.getsize(target) / 1024.0
                print(f"[SUCCESS] Built {os.path.basename(target)} ({size_kb:.1f} KB)")
            else:
                print(f"[ERROR] Target {target} was not created!")
                sys.exit(1)

    print("=" * 70)
    print("NATIVE C++ DWA DUAL-TARGET BUILD COMPLETE (SIMD + SCALAR)")
    print("=" * 70)


if __name__ == "__main__":
    build_cpp_backend()
