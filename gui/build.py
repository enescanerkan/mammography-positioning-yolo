#!/usr/bin/env python3
"""
Build Script for Mammogram Positioning Analysis

This script automates the process of building a standalone Windows executable
using PyInstaller.

Usage:
    python build.py
    
Requirements:
    pip install pyinstaller
"""

import subprocess
import sys
import os
from pathlib import Path


def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"✓ PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("✗ PyInstaller not found")
        print("  Install with: pip install pyinstaller")
        return False


def clean_build():
    """Clean previous build artifacts."""
    import shutil
    
    dirs_to_clean = ['build', 'dist']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"  Cleaning {dir_name}/...")
            shutil.rmtree(dir_name)
    
    print("✓ Build directories cleaned")


def build_exe():
    """Build the executable using PyInstaller."""
    spec_file = "mammogram_analysis.spec"
    
    if not os.path.exists(spec_file):
        print(f"✗ Spec file not found: {spec_file}")
        return False
    
    print("Building executable...")
    print("-" * 50)
    
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", spec_file, "--clean"],
        capture_output=False
    )
    
    if result.returncode == 0:
        exe_path = Path("dist") / "MammogramAnalysis.exe"
        if exe_path.exists():
            size_mb = exe_path.stat().st_size / (1024 * 1024)
            print("-" * 50)
            print(f"✓ Build successful!")
            print(f"  Output: {exe_path}")
            print(f"  Size: {size_mb:.1f} MB")
            return True
    
    print("✗ Build failed")
    return False


def main():
    """Main build process."""
    print("=" * 50)
    print("Mammogram Analysis - Build Script")
    print("=" * 50)
    print()
    
    # Check requirements
    print("[1/3] Checking requirements...")
    if not check_pyinstaller():
        sys.exit(1)
    print()
    
    # Clean previous builds
    print("[2/3] Cleaning previous builds...")
    clean_build()
    print()
    
    # Build
    print("[3/3] Building executable...")
    if not build_exe():
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("Build complete!")
    print()
    print("Note: The weights folder is NOT included in the exe.")
    print("Users will be prompted to download weights on first run.")
    print("=" * 50)


if __name__ == "__main__":
    main()
