# ⟡⟡⟡ Σιγναλ Σιγμα ⟡⟡⟡

## Synopsis

**TODO** Write this paragraph

## Requirements

- Python 3.11.3
- pyenv

## Installation

1. Navigate to a working directory of your choice, then clone the repository and enter it:

   ``` shell
   git clone https://github.com/julialoeschel/capstone-SignalSigma.git &&
       cd capstone-SignalSigma
   ```

2. Choose a setup option based on your operating system and intended use:

   - `make basic-unix` / `make basic-win`: for general use or exploration (core dependencies only).
   - `make dev-unix` / `make dev-win`: for contributors (includes development tools like linters and pre-commit hooks).

   Details for each Makefile target are provided [below](#makefile-targets).

3. Activate the virtual environment:

   - On macOS/Linux, run:

     ```shell
     source .venv/bin/activate
     ```

   - On Windows (PowerShell), run:

     ``` powershell
     .\.venv\Scripts\Activate.ps1
     ```

## Makefile Targets

**TODO:** Proof-check the Windows versions
**TODO:** Add explanations for clearing and resetting

Below, we explain what each Makefile target does. You can also run the corresponding commands manually if you prefer.

### Basic

- `make basic-unix`

  ```shell
  pyenv local 3.11.3
  python -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install .
  ```

- `make basic-win`

  ``` powershell
  pyenv local 3.11.3
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install .
  ```

### Dev

- `make dev-unix`

  ``` shell
  pyenv local 3.11.3
  python -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install .[dev]
  .venv/bin/pre-commit install
  ```

- `make dev-win`

  ``` powershell
  pyenv local 3.11.3
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install .[dev]
  .\.venv\Scripts\pre-commit.exe install
  ```
