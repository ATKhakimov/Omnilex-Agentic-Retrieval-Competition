#!/usr/bin/env python3
"""Validate submission file format.

Usage:
    python scripts/validate_submission.py submission.csv
    python scripts/validate_submission.py submission.csv --verbose
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.evaluation.scorer import validate_submission_format


def main():
    parser = argparse.ArgumentParser(description="Validate submission file format")
    parser.add_argument(
        "submission",
        type=Path,
        help="Path to submission CSV file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed validation output",
    )

    args = parser.parse_args()

    print(f"Validating: {args.submission}")
    print("-" * 50)

    # Run format validation
    errors = validate_submission_format(args.submission)

    if errors:
        print("VALIDATION FAILED")
        print()
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Format validation: PASSED")

    print()
    print("Validation complete!")


if __name__ == "__main__":
    main()
