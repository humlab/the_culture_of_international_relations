#!/usr/bin/env python3
"""
Setup script optimized for uv to add the project root to Python path.
Run this before running any modules to ensure imports work correctly.
"""
import sys
import os
import subprocess


def setup_environment():
    """Setup the development environment"""
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Add the project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"Added {project_root} to Python path")

    # Install package in development mode with uv if not already installed
    try:
        import importlib.util

        spec = importlib.util.find_spec("the_culture_of_international_relations")
        if spec is None:
            print("Installing package in development mode with uv...")
            result = subprocess.run(
                ["uv", "pip", "install", "-e", "."],
                cwd=project_root,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("Package installed successfully!")
            else:
                print(f"Error installing package: {result.stderr}")
    except Exception as e:
        print(f"Could not check/install package: {e}")

    print("Current Python path (first 5 entries):")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")


if __name__ == "__main__":
    setup_environment()
