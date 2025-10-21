"""
Master test runner - runs all tests sequentially.
Run: python tests/run_all_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Test definitions
TESTS = [
    {
        "name": "Phase 1: Modal Basic",
        "command": ["modal", "run", "tests/test_modal_basic.py"],
        "cost": "$0.01",
        "time": "2 min",
        "required": True,
    },
    {
        "name": "Phase 2: Modal Distributed",
        "command": ["modal", "run", "tests/test_modal_distributed.py"],
        "cost": "$0.20",
        "time": "5 min",
        "required": True,
    },
    {
        "name": "Phase 3: Modal WandB",
        "command": ["modal", "run", "tests/test_modal_wandb.py"],
        "cost": "$0.20",
        "time": "3 min",
        "required": True,
    },
    {
        "name": "Phase 4: Modal FSDP",
        "command": ["modal", "run", "tests/test_modal_fsdp.py"],
        "cost": "$0.30",
        "time": "5 min",
        "required": True,
    },
    {
        "name": "Phase 5: Full Smoke Test",
        "command": ["modal", "run", "tests/test_modal_full_smoke.py"],
        "cost": "$1-2",
        "time": "15 min",
        "required": False,  # Optional - expensive
    },
]


def run_test(test_config, test_num, total_tests):
    """Run a single test and return success status."""
    print("\n" + "="*80)
    print(f"Test {test_num}/{total_tests}: {test_config['name']}")
    print(f"Cost: {test_config['cost']}, Time: ~{test_config['time']}")
    print("="*80)

    # Confirmation for expensive tests
    if test_config["cost"] != "FREE" and not test_config["required"]:
        response = input(f"\n⚠️  This test costs {test_config['cost']}. Continue? (y/n): ")
        if response.lower() != 'y':
            print("⏭️  Skipping...")
            return None

    start_time = time.time()

    try:
        result = subprocess.run(
            test_config["command"],
            check=True,
            capture_output=False,
            text=True,
        )
        elapsed = time.time() - start_time
        print(f"\n✅ PASSED in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return None


def main():
    """Run all tests sequentially."""
    print("\n" + "="*80)
    print("ManimBOT Master Test Suite")
    print("="*80)
    print(f"\nTotal tests: {len(TESTS)}")
    print(f"Estimated time: ~25 minutes")
    print(f"Estimated cost: ~$0.71 (required) + $1-2 (optional smoke test)")
    print("\n" + "="*80)

    response = input("\nRun all tests? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Track results
    results = []
    total_start = time.time()

    # Run each test
    for i, test in enumerate(TESTS, 1):
        result = run_test(test, i, len(TESTS))

        if result is None:
            # Skipped or interrupted
            results.append({"name": test["name"], "status": "SKIPPED"})
            if i < len(TESTS):
                response = input("\nContinue to next test? (y/n): ")
                if response.lower() != 'y':
                    print("\nStopping test suite.")
                    break
        elif result is False:
            # Failed
            results.append({"name": test["name"], "status": "FAILED"})
            print(f"\n❌ {test['name']} failed!")

            if test["required"]:
                print("⚠️  This is a required test. Cannot proceed.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("\nStopping test suite.")
                    break
        else:
            # Passed
            results.append({"name": test["name"], "status": "PASSED"})

    # Print summary
    total_elapsed = time.time() - total_start

    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for result in results:
        status_symbol = {
            "PASSED": "✅",
            "FAILED": "❌",
            "SKIPPED": "⏭️ ",
        }[result["status"]]
        print(f"{status_symbol} {result['name']}: {result['status']}")

    passed = sum(1 for r in results if r["status"] == "PASSED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")

    print("\n" + "-"*80)
    print(f"Total: {len(results)} tests")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print("="*80)

    # Final verdict
    if failed == 0 and passed >= 4:  # At least 4 core tests passed
        print("\n🎉 ALL CORE TESTS PASSED!")
        print("✅ Your training setup is ready!")
        print("\nNext step:")
        print("  modal run src.utils.modal.modal_app::train")
    elif failed > 0:
        print("\n⚠️  SOME TESTS FAILED")
        print("❌ Fix failing tests before running full training.")
    else:
        print("\n⚠️  NOT ENOUGH TESTS RUN")
        print("Run at least phases 1-4 to verify setup.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test suite interrupted by user")
        sys.exit(1)
