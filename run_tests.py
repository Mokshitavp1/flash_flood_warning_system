#!/usr/bin/env python3
"""Custom Test Runner for HydroAlert Suite."""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print("\n" + "=" * 70)
    print("RUNNING HYDROALERT UNIT & INTEGRATION TEST SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"TESTS RUN: {result.testsRun}")
    print(f"FAILURES: {len(result.failures)}")
    print(f"ERRORS: {len(result.errors)}")
    print(f"SKIPPED: {len(result.skipped)}")
    print("=" * 70)

    if not result.wasSuccessful():
        sys.exit(1)
    print("ALL TESTS PASSED SUCCESSFULLY! NFR1 DATA INTEGRITY VERIFIED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
