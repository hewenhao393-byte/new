import argparse

from ablation_analysis.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run paired 43-versus-40 feature ablations")
    parser.add_argument("--features", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_pipeline(arguments.features, arguments.baseline, arguments.output)


if __name__ == "__main__":
    main()
