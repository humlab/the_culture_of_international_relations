#!/bin/bash
# setup_env.sh - Set up environment for the project

# Add current directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo "Added $(pwd) to PYTHONPATH"
echo "Current PYTHONPATH: $PYTHONPATH"

# Install the package in development mode using uv
# This allows the package to be imported from anywhere
uv pip install -e .