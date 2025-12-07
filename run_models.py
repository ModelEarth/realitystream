import argparse
import os
import sys
import textwrap

def main():
    parser = argparse.ArgumentParser(
        description="Run RealityStream models using a parameters.yaml file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python run_models.py parameters/parameters.yaml
              python run_models.py path/to/custom-params.yaml
            """
        ),
    )
    parser.add_argument(
        "yaml",
        help="Path to parameters.yaml (relative or absolute)",
    )
    args = parser.parse_args()

    params_path = os.path.abspath(args.yaml)

    if not os.path.exists(params_path):
        print(f"[ERROR] parameters file not found: {params_path}")
        sys.exit(1)

    print("[INFO] RealityStream local runner")
    print(f"[INFO] Using parameters file: {params_path}")
    print()
    print("TODO: integrate this CLI with the Run-Models-bkup.ipynb pipeline.")
    print("For now, this verifies the path and prepares the interface.")
    print()
    print("Next step (to be implemented in a follow-up PR):")
    print("- Use papermill or a direct Python script to execute the ML run")
    print("- Save outputs to the report folder as described in the docs")

if __name__ == "__main__":
    main()
