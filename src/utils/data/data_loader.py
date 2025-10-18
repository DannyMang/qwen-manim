from datasets import load_dataset, interleave_datasets

def load_all_datasets(streaming=True):
    """
    Load and unify all 4 manim datasets into a single stream.

    Total: 7,622 training examples
    - generaleoley/manim-codegen: 1,622 examples (medium complexity)
    - bespokelabs/bespoke-manim: 1,000 examples (highest quality, 99.5% validated)
    - thanhkt/manim_code: 4,400 examples (largest dataset)
    - Edoh/manim_python: 599 examples (basic, concise)

    Args:
        streaming: If True, use streaming mode (recommended for Modal GPUs)

    Returns:
        Unified dataset with schema: {prompt, code, source}
    """

    # Dataset 1: generaleoley/manim-codegen (1,622 examples)
    ds1 = load_dataset("generaleoley/manim-codegen", split="train", streaming=streaming)
    ds1 = ds1.map(lambda x: {
        "prompt": x["query"],
        "code": x["answer"],
        "source": "generaleoley"
    })
    ds1 = ds1.remove_columns([col for col in ds1.column_names if col not in ["prompt", "code", "source"]])

    # Dataset 2: bespokelabs/bespoke-manim (1,000 examples)
    ds2 = load_dataset("bespokelabs/bespoke-manim", split="train", streaming=streaming)
    ds2 = ds2.map(lambda x: {
        "prompt": x["question"],  # Short question (or use x["narration"] for detailed script)
        "code": x["python_code"],
        "source": "bespokelabs"
    })
    ds2 = ds2.remove_columns([col for col in ds2.column_names if col not in ["prompt", "code", "source"]])

    # Dataset 3: thanhkt/manim_code (4,400 examples)
    ds3 = load_dataset("thanhkt/manim_code", split="train", streaming=streaming)
    ds3 = ds3.map(lambda x: {
        "prompt": x["input"],
        "code": x["output"],
        "source": "thanhkt"
    })
    ds3 = ds3.remove_columns([col for col in ds3.column_names if col not in ["prompt", "code", "source"]])

    # Dataset 4: Edoh/manim_python (599 )
    ds4 = load_dataset("Edoh/manim_python", split="train", streaming=streaming)
    ds4 = ds4.map(lambda x: {
        "prompt": x["instruction"],
        "code": x["output"],
        "source": "edoh"
    })
    ds4 = ds4.remove_columns([col for col in ds4.column_names if col not in ["prompt", "code", "source"]])

    # Combine all datasets with interleaving for streaming or concatenation for non-streaming
    if streaming:
        combined = interleave_datasets([ds1, ds2, ds3, ds4])
    else:
        from datasets import concatenate_datasets
        combined = concatenate_datasets([ds1, ds2, ds3, ds4])

    return combined


def load_test_dataset(streaming=True):
    """
    Load the test split (only available from Edoh/manim_python).

    Returns:
        Test dataset with 51 examples, schema: {prompt, code, source}
    """
    test_ds = load_dataset("Edoh/manim_python", split="test", streaming=streaming)
    test_ds = test_ds.map(lambda x: {
        "prompt": x["instruction"],
        "code": x["output"],
        "source": "edoh_test"
    })
    test_ds = test_ds.remove_columns([col for col in test_ds.column_names if col not in ["prompt", "code", "source"]])

    return test_ds
