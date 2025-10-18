from datasets import load_dataset, interleave_datasets
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist

def load_all_datasets(streaming=True):
    """
    Load and unify all 4 manim datasets into a single stream.

    Total: 7,622 training examples
    - generaleoley/manim-codegen: 1,622 examples
    - bespokelabs/bespoke-manim: 1,000 examples
    - thanhkt/manim_code: 4,400 examples
    - Edoh/manim_python: 599 examples

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

class StreamingManimDataset(IterableDataset):
    """
    Wraps huggingface streaming dataset for PyTorch dataloader with automatic distributed sharding
    """

    def __init__(self, hf_dataset, tokenizer, max_length=2048):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # Single GPU fallback
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        for idx, example in enumerate(self.dataset):
            if idx % self.world_size != self.rank:
                continue

            text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['code']}"

            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            yield{
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": encoded["input_ids"].squeeze(0),
            }

def get_dataloader(tokenizer, batch_size=4, max_length=2048, streaming=True, num_workers=0):
    """
    Get PyTorch dataloader with automatic sharding for FSDP.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size per GPU
        max_length: Max sequence length
        streaming: Whether to use streaming mode
        num_workers: Number of data loading workers (must be 0 for streaming)

    Returns:
        PyTorch DataLoader with automatic rank-based sharding
    """
    hf_dataset = load_all_datasets(streaming=streaming)
    dataset = StreamingManimDataset(hf_dataset, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
