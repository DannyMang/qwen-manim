"""
Test basic Modal deployment without GPU.
Run: modal run tests/test_modal_basic.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.0",
    )
)

app = modal.App("manimbot-test-basic", image=image)


@app.function()
def test_basic_imports():
    """Test that all basic imports work on Modal."""
    print("Testing imports on Modal...")

    import torch
    print(f"✅ torch {torch.__version__}")

    import transformers
    print(f"✅ transformers {transformers.__version__}")

    import yaml
    print(f"✅ PyYAML")

    from dotenv import load_dotenv
    print(f"✅ python-dotenv")

    print("\n🎉 Basic Modal test passed!")
    return "success"


@app.local_entrypoint()
def main():
    """Run the basic test."""
    result = test_basic_imports.remote()
    print(f"\nResult: {result}")
