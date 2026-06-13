"""Cross-subject LOSO orchestration entry reusing the default ML pipeline."""

from datetime import datetime

from machine_learning_process import parse_args, run_experiment


def build_loso_args():
    """Create the standard cross-subject LOSO experiment arguments."""
    args = parse_args()
    args.validation_strategy = "loso"
    args.run_name = datetime.now().strftime("course_report_loso_%Y%m%d_%H%M%S")
    return args


def main():
    """Run the fixed Leave-One-Subject-Out experiment."""
    run_experiment(build_loso_args())


if __name__ == "__main__":
    main()
