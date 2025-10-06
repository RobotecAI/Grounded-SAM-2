"""
Unit test for package building and installation.
Run with: python test_package_build.py
"""
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run shell command and return result."""
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


def test_package_build_and_install():
    """Test that package builds and installs correctly."""
    print("Testing package build and installation...\n")
    
    # Get repo root (parent of tests/ directory)
    repo_root = Path(__file__).parent.parent
    
    # Step 1: Clean previous builds
    print("1. Cleaning previous builds...")
    for path in ["dist", "build", "*.egg-info"]:
        for item in repo_root.glob(path):
            if item.is_dir():
                shutil.rmtree(item)
    print("   Cleaned\n")
    
    # Step 2: Build package
    print("2. Building package...")
    returncode, stdout, stderr = run_command("python -m build", cwd=repo_root)
    if returncode != 0:
        print(f"   ❌ Build failed:")
        print(f"   stdout: {stdout}")
        print(f"   stderr: {stderr}")
        return False
    print("   Package built successfully")
    if stdout:
        print(f"   {stdout.strip()}")
    print()
    
    # Step 3: Check dist files exist
    print("3. Checking distribution files...")
    dist_dir = repo_root / "dist"
    if not dist_dir.exists():
        print("   ❌ dist/ directory not found")
        return False
    
    whl_files = list(dist_dir.glob("*.whl"))
    tar_files = list(dist_dir.glob("*.tar.gz"))
    
    if not whl_files:
        print("   ❌ No .whl file found")
        return False
    if not tar_files:
        print("   ❌ No .tar.gz file found")
        return False
    
    print(f"   Found: {whl_files[0].name}")
    print(f"   Found: {tar_files[0].name}\n")
    
    # Step 4: Test installation in virtual environment
    print("4. Testing installation in temp venv...")
    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "test_venv"
        
        # Create venv
        returncode, _, stderr = run_command(f"python -m venv {venv_path}")
        if returncode != 0:
            print(f"   ❌ venv creation failed:\n{stderr}")
            return False
        
        # Get pip path
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = venv_path / "Scripts" / "pip.exe"  # Windows
        
        # Install package
        whl_path = whl_files[0].absolute()
        returncode, _, stderr = run_command(f"{pip_path} install {whl_path}")
        if returncode != 0:
            print(f"   ❌ Installation failed:\n{stderr}")
            return False
        
        print("   Package installed successfully\n")
        
        # Step 5: Verify import
        print("5. Verifying package import...")
        python_path = venv_path / "bin" / "python"
        if not python_path.exists():
            python_path = venv_path / "Scripts" / "python.exe"  # Windows
        
        returncode, stdout, stderr = run_command(
            f"{python_path} -c 'import sam2; print(\"sam2 imported successfully\")'")
        
        if returncode != 0:
            print(f"   ❌ Import failed:\n{stderr}")
            return False
        
        print(f"   {stdout.strip()}\n")
    
    print("All tests passed! Package is ready for publishing.\n")
    return True


if __name__ == "__main__":
    try:
        success = test_package_build_and_install()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        sys.exit(1)

