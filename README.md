# The Culture of International Relations

Repository for NLP-scripts related to the The Culture of International Relations project.

## Installation (macOS)

This is a legacy system that was previously deployed using Jupyter Hub. For local installation on macOS, follow these steps:

### Prerequisites

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install uv** (Python package manager):
   ```bash
   brew install uv
   ```

3. **Install Python 3.11+** (if not already available):
   ```bash
   brew install python@3.11
   ```

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/humlab/the_culture_of_international_relations.git
   cd the_culture_of_international_relations
   ```

2. **Create and activate a virtual environment with uv**:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```
   
   This will install all required packages including:
   - Jupyter Lab for notebook interface
   - spaCy with English language model
   - Network analysis tools (networkx, python-louvain)
   - Text processing libraries (textacy, nltk)
   - Visualization tools (bokeh, matplotlib, seaborn)

4. **Install development dependencies** (optional, for contributors):
   ```bash
   uv pip install --group dev
   ```

5. **Setup the Python path** (run before working with notebooks):
   ```bash
   python setup_env_uv.py
   ```

### Running Jupyter Lab

Once installation is complete, launch Jupyter Lab:

```bash
uv run jupyter lab
```

This will open the Jupyter interface in your default web browser where you can access notebooks in the `notebooks/` directory.

### Project Structure

- `common/` - Shared utilities and configuration
- `notebooks/` - Jupyter notebooks organized by analysis type:
  - `bens_prioritized_methods/`
  - `network_analysis/`
  - `publications/`
  - `quantitative_analysis/`
  - `text_analysis/`
- `data/` - Data files and resources
- `legacy/` - Legacy scripts (deprecated)

### Development Tools

The project includes several make commands for code quality:

```bash
make lint      # Run linting tools
make tidy      # Format code with black and isort
make ruff      # Check code with ruff
```

### Troubleshooting

- **Import errors**: Make sure you've run `python setup_env_uv.py` to add the project root to your Python path
- **Missing spaCy model**: The English language model should install automatically, but if needed you can install it manually:
  ```bash
  uv run python -m spacy download en_core_web_lg
  ```
- **Dependency conflicts**: Try removing the virtual environment and recreating it:
  ```bash
  rm -rf .venv
  uv venv
  source .venv/bin/activate
  uv pip install -e .
  ```

