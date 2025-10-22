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
    ds1 = ds1.map(
        lambda x: {
            "prompt": x["query"],
            "code": x["answer"],
            "source": "generaleoley"
        },
        remove_columns=["query", "answer"] if streaming else None
    )

    # Dataset 2: bespokelabs/bespoke-manim (1,000 examples)
    ds2 = load_dataset("bespokelabs/bespoke-manim", split="train", streaming=streaming)
    ds2 = ds2.map(
        lambda x: {
            "prompt": x["question"],  # Short question (or use x["narration"] for detailed script)
            "code": x["python_code"],
            "source": "bespokelabs"
        },
        remove_columns=[
            "subject", "topic", "question", "title", "narration", "visual_elements",
            "equations", "key_timestamps", "visual_style", "concept_id", "python_code",
            "scene_class_name", "generation_time", "filename", "message", "error",
            "stdout", "stderr", "video"
        ] if streaming else None
    )

    # Dataset 3: thanhkt/manim_code (4,400 examples)
    ds3 = load_dataset("thanhkt/manim_code", split="train", streaming=streaming)
    ds3 = ds3.map(
        lambda x: {
            "prompt": x["input"],
            "code": x["output"],
            "source": "thanhkt"
        },
        remove_columns=["input", "output"] if streaming else None
    )

    # Dataset 4: Edoh/manim_python (599 )
    ds4 = load_dataset("Edoh/manim_python", split="train", streaming=streaming)
    ds4 = ds4.map(
        lambda x: {
            "prompt": x["instruction"],
            "code": x["output"],
            "source": "edoh"
        },
        remove_columns=["instruction", "output"] if streaming else None
    )

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
    test_ds = test_ds.map(
        lambda x: {
            "prompt": x["instruction"],
            "code": x["output"],
            "source": "edoh_test"
        },
        remove_columns=["instruction", "output"] if streaming else None
    )

    return test_ds

class StreamingManimDataset(IterableDataset):
    """
    Wraps huggingface streaming dataset for PyTorch dataloader with automatic distributed sharding
    """

    def __init__(self, hf_dataset, tokenizer, max_length=2048, total_examples=7622):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.total_examples = total_examples

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # Single GPU fallback
            self.rank = 0
            self.world_size = 1

    def __len__(self):
        """
        Return approximate number of examples for this rank.
        Since we use modulo-based sharding, each rank gets ~total_examples/world_size
        """
        return self.total_examples // self.world_size

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

def get_dataloader(tokenizer, batch_size=4, max_length=2048, streaming=True, num_workers=0, total_examples=7622):
    """
    Get PyTorch dataloader with automatic sharding for FSDP.

    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size per GPU
        max_length: Max sequence length
        streaming: Whether to use streaming mode
        num_workers: Number of data loading workers (must be 0 for streaming)
        total_examples: Total number of examples in the dataset (default 7622)

    Returns:
        PyTorch DataLoader with automatic rank-based sharding
    """
    hf_dataset = load_all_datasets(streaming=streaming)
    dataset = StreamingManimDataset(hf_dataset, tokenizer, max_length, total_examples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader
