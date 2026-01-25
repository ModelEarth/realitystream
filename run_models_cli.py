import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_models_cli.py <path-or-url-to-parameters.yaml>")
        sys.exit(1)

    params_arg = sys.argv[1]

    # Handle local path vs URL
    if params_arg.startswith("http://") or params_arg.startswith("https://"):
        params_path = params_arg
    else:
        params_path = str(Path(params_arg).expanduser().resolve())

    # Set as environment variable for the notebook
    env = os.environ.copy()
    env["PARAMETERS_YAML_PATH"] = params_path

    notebook = "models/Run-Models-bkup.ipynb"
    output_nb = "Run-Models-output.ipynb"

    cmd = [
        "jupyter",
        "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=0",
        "--output", output_nb,
        notebook,
    ]

    print(f"Running {notebook} with PARAMETERS_YAML_PATH={params_path}")
    subprocess.run(cmd, check=True, env=env)
    print(f"✅ Done! Executed notebook saved as {output_nb}")

if __name__ == "__main__":
    main()
