"""
Conversational dataset pipeline for Continuum SLM training.

Downloads and processes conversational datasets:
1. Stanford Alpaca (tatsu-lab/alpaca) - 52K instruction pairs (PRIMARY - works reliably)
2. Custom JSONL data via load_from_jsonl()

NOTE (2026): HuggingFace datasets v4+ broke support for script-based datasets.
OpenAssistant (oasst1) and Dolly (databricks-dolly-15k) use legacy CDN
infrastructure (Xet) that generates invalid signed URLs, causing 403 errors.
Only Parquet-based datasets like Alpaca work reliably.

Outputs formatted conversation data ready for tokenization and training.
"""

import os
import json
import random
from typing import List, Dict, Optional, Iterator, Tuple
from tqdm import tqdm

from continuum.conversation.template import ChatTemplate, Message, Role


# ============================================================================
# Dataset Downloaders
# ============================================================================

def get_openassistant_data(split: str = "train", max_samples: Optional[int] = None) -> List[Dict]:
    """
    DISABLED - HuggingFace CDN infrastructure broken (2025-2026).
    
    OpenAssistant uses the legacy Xet CDN that generates invalid signed URLs.
    Returns empty list. Use Alpaca (52K) or custom JSONL data instead.
    """
    print("  ⚠️ OpenAssistant is DISABLED - HuggingFace CDN infrastructure broken (2025-2026)")
    print("  ✅ Use Alpaca (52K instructions) or dataset.load_from_jsonl() instead.")
    return []


def get_dolly_data(max_samples: Optional[int] = None) -> List[Dict]:
    """
    DISABLED - Same HuggingFace CDN infrastructure issue as OpenAssistant.
    
    Dolly uses the same broken Xet CDN that generates invalid signed URLs.
    Returns empty list. Use Alpaca (52K) or custom JSONL data instead.
    """
    print("  ⚠️ Dolly is DISABLED - same HuggingFace CDN infrastructure issue")
    print("  ✅ Use Alpaca (52K instructions) or dataset.load_from_jsonl() instead.")
    return []


def get_alpaca_data(max_samples: Optional[int] = None) -> List[Dict]:
    """
    Download Stanford Alpaca dataset (52K instructions).
    
    Public dataset using proper Parquet format - works reliably.
    No authentication needed.
    Each item is an instruction -> response pair.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to download conversational datasets")
    
    print("Loading Stanford Alpaca (52K instructions)...")
    
    try:
        import logging
        old_level = logging.getLogger("huggingface_hub").getEffectiveLevel()
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        logging.getLogger("huggingface_hub").setLevel(old_level)
    except Exception:
        print("  ⚠️ Alpaca download failed. Skipping.")
        return []
    
    conversations = []
    for item in ds:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "") or ""
        response = item.get("output", "")
        
        if not instruction or not response:
            continue
        
        # Build conversation - include input context if present
        if input_text:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction
        
        conversations.append({
            "conversation": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
        })
    
    if max_samples:
        conversations = conversations[:max_samples]
    
    print(f"  Loaded {len(conversations)} conversations")
    return conversations


def get_tulu_data(max_samples: Optional[int] = None) -> List[Dict]:
    """
    Download Allen AI Tulu 2 dataset (mixture of instruction datasets).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets to download conversational datasets")
    
    print("Loading Allen AI Tulu 2...")
    
    try:
        ds = load_dataset("allenai/tulu-2-dataset", split="train")
    except Exception as e:
        print(f"  ❌ Could not load Tulu 2: {e}")
        print("  Skipping Tulu 2 dataset.")
        return []
    
    conversations = []
    for item in ds:
        messages = item.get("messages", [])
        if not messages:
            continue
        
        conv = []
        for m in messages:
            content = m.get("content", "") or ""
            if not content:
                continue
            conv.append({
                "role": m.get("role", "user"),
                "content": content,
            })
        
        if len(conv) >= 2:
            conversations.append({"conversation": conv})
    
    if max_samples:
        conversations = conversations[:max_samples]
    
    print(f"  Loaded {len(conversations)} conversations")
    return conversations


# ============================================================================
# Dataset Builder
# ============================================================================

class ConversationalDataset:
    """
    Builds a tokenized dataset from conversational data.
    
    Downloads, formats, tokenizes, and saves conversational datasets
    for training Continuum SLM.
    
    Usage:
        dataset = ConversationalDataset(tokenizer)
        dataset.load_from_hub(max_samples=50000)
        # or: dataset.load_from_jsonl("data.jsonl")
        loader = dataset.get_dataloader(batch_size=8)
    """
    
    def __init__(self, tokenizer, max_seq_len: int = 1024):
        """
        Args:
            tokenizer: ContinuumTokenizer instance
            max_seq_len: Maximum sequence length after tokenization
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []  # (input_ids, labels)
        self.chat_template = ChatTemplate(add_generation_prompt=False)
    
    def _format_and_tokenize(self, conversation: List[Dict], mask_non_assistant: bool = True) -> List[Tuple[List[int], List[int]]]:
        """
        Format a conversation and tokenize it.
        
        Returns list of (input_ids, labels) pairs, potentially
        splitting long conversations into multiple chunks.
        
        If mask_non_assistant=True, loss is only computed for
        assistant response tokens (not user/system prompts).
        """
        # Convert to Message objects
        messages = []
        for turn in conversation:
            try:
                role = Role(turn["role"])
                messages.append(Message(role, turn["content"]))
            except Exception:
                continue
        
        if not messages:
            return []
        
        # Format full conversation
        formatted = ChatTemplate(add_generation_prompt=False).format_messages(messages)
        
        if not formatted or not formatted.strip():
            return []
        
        # Tokenize
        tokens = self.tokenizer.encode_with_special(
            formatted, add_bos=True, add_eos=True
        )
        
        if not tokens or len(tokens) < 5:
            return []
        
        # Split into chunks if too long
        chunks = []
        max_len = self.max_seq_len
        
        if len(tokens) <= max_len:
            # Single chunk
            input_ids = tokens[:-1]
            labels = tokens[1:]
            if mask_non_assistant:
                labels = self._mask_non_assistant_tokens(labels, input_ids)
            chunks.append((input_ids, labels))
        else:
            # Split into overlapping chunks
            stride = max_len // 2
            for start in range(0, len(tokens) - 1, stride):
                end = min(start + max_len, len(tokens))
                chunk = tokens[start:end]
                if len(chunk) < 10:  # Too short, skip
                    break
                input_ids = chunk[:-1]
                labels = chunk[1:]
                if mask_non_assistant:
                    labels = self._mask_non_assistant_tokens(labels, input_ids)
                chunks.append((input_ids, labels))
                if end == len(tokens):
                    break
        
        return chunks
    
    def _mask_non_assistant_tokens(self, labels: List[int], tokens: List[int]) -> List[int]:
        """
        Mask loss for non-assistant tokens so the model only learns
        to predict assistant responses (not user/system prompts).
        
        Tokens before the first <|assistant|> token are set to -100 (ignore).
        """
        # Find all occurrences of assistant token
        assistant_pattern = self.chat_template.ASSISTANT_TOKEN
        assistant_ids = self.tokenizer.encode(assistant_pattern)
        
        if not assistant_ids:
            return labels
        
        # Simple approach: find positions where assistant starts
        masked = [-100] * len(labels)
        
        # Find positions where assistant response begins
        i = 0
        in_assistant = False
        while i < len(tokens):
            # Check if this position matches assistant token
            if i + len(assistant_ids) <= len(tokens):
                if tokens[i:i+len(assistant_ids)] == assistant_ids:
                    in_assistant = True
                    i += len(assistant_ids)
                    continue
            
            if in_assistant:
                if i < len(labels):
                    # Check for end token or next user token
                    if tokens[i] == self.tokenizer.eos_id:
                        in_assistant = False
                        if i < len(labels):
                            masked[i] = labels[i]
                    else:
                        masked[i] = labels[i]
            
            i += 1
        
        return masked
    
    def add_sample(self, conversation: List[Dict]):
        """Add a single conversation to the dataset."""
        chunks = self._format_and_tokenize(conversation)
        self.samples.extend(chunks)
    
    def load_from_jsonl(self, path: str, max_samples: Optional[int] = None):
        """
        Load conversations from a JSONL file.
        
        Each line should be a JSON object with:
        - "conversation": list of {"role": "user"|"assistant", "content": "..."}
        """
        print(f"Loading conversations from {path}...")
        count = 0
        with open(path, "r") as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "conversation" in data:
                    self.add_sample(data["conversation"])
                    count += 1
        
        print(f"  Loaded {count} conversations → {len(self.samples)} training samples")
    
    def load_from_hub(
        self,
        include_oasst1: bool = False,
        include_dolly: bool = False,
        include_alpaca: bool = True,
        include_tulu: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Load datasets from HuggingFace hub.
        
        Args:
            include_oasst1: Load OpenAssistant (DISABLED - HF CDN broken)
            include_dolly: Load Databricks Dolly 2.0 (DISABLED - same CDN issue)
            include_alpaca: ✅ Load Stanford Alpaca (public, 52K - WORKS!)
            include_tulu: Load Tulu 2
            max_samples: Maximum total samples
        """
        print("=" * 50)
        print("📚 DOWNLOADING DATASETS")
        print("=" * 50)
        print()
        
        all_conversations = []
        remaining = max_samples
        
        if include_oasst1:
            data = get_openassistant_data("train", remaining)
            all_conversations.extend(data)
            remaining = max_samples - len(all_conversations) if max_samples else None
            print()
        
        if include_alpaca and (remaining is None or remaining > 0):
            data = get_alpaca_data(remaining)
            all_conversations.extend(data)
            remaining = max_samples - len(all_conversations) if max_samples else None
            print()
        
        if include_dolly and (remaining is None or remaining > 0):
            data = get_dolly_data(remaining)
            all_conversations.extend(data)
            print()
        
        if include_tulu and (remaining is None or remaining > 0):
            data = get_tulu_data(remaining)
            all_conversations.extend(data)
            print()
        
        if not all_conversations:
            print("❌ No datasets could be loaded!")
            print("   Please check your internet connection.")
            return
        
        print(f"\n📝 Tokenizing {len(all_conversations)} conversations...")
        for item in tqdm(all_conversations):
            self.add_sample(item["conversation"])
        
        print(f"\n✅ Dataset ready!")
        print(f"   → {len(self.samples)} training samples created")
    
    def save(self, path: str):
        """Save tokenized dataset to disk."""
        import torch
        data = {
            "input_ids": [s[0] for s in self.samples],
            "labels": [s[1] for s in self.samples],
            "metadata": {
                "max_seq_len": self.max_seq_len,
                "num_samples": len(self.samples),
            }
        }
        torch.save(data, path)
        print(f"Dataset saved to {path} ({len(self.samples)} samples)")
    
    @classmethod
    def load(cls, path: str, tokenizer) -> "ConversationalDataset":
        """Load tokenized dataset from disk."""
        import torch
        data = torch.load(path, weights_only=False)
        ds = cls(tokenizer, max_seq_len=data["metadata"]["max_seq_len"])
        ds.samples = list(zip(data["input_ids"], data["labels"]))
        print(f"Dataset loaded from {path} ({len(ds.samples)} samples)")
        return ds
    
    def get_dataloader(self, batch_size: int = 8, shuffle: bool = True):
        """Create a standard PyTorch DataLoader (pads to batch max)."""
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset
        
        class _Dataset(Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                input_ids, labels = self.samples[idx]
                return (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )
        
        def collate_fn(batch):
            input_ids = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # Pad to same length
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            return {
                "input_ids": input_ids_padded,
                "labels": labels_padded,
            }
        
        return DataLoader(
            _Dataset(self.samples),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )
    
    def get_bucket_dataloader(
        self,
        batch_size: int = 8,
        num_buckets: int = 8,
        shuffle: bool = True,
        subset_indices = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """
        ⚡ Phase 1: Bucket sampler — groups same-length sequences to minimize padding waste.
        
        Instead of padding all sequences in a batch to the longest one,
        groups sequences into length buckets (±5% range) so each batch
        has uniform-length sequences → near-zero padding waste.
        
        Expected: 15-25% less wasted compute on padding tokens.
        
        Args:
            batch_size: Samples per batch
            num_buckets: Number of length buckets (more = tighter grouping)
            shuffle: Shuffle within buckets
            num_workers: DataLoader workers
            pin_memory: Pin memory for faster GPU transfer
            prefetch_factor: Prefetch batches
            persistent_workers: Keep workers alive across epochs
        
        Returns:
            DataLoader with bucket-grouped batches
        """
        import torch
        from torch.nn.utils.rnn import pad_sequence
        from torch.utils.data import DataLoader, Dataset, Sampler
        
        # Handle subset (for train/val split)
        if subset_indices is not None:
            active_samples = [self.samples[i] for i in subset_indices]
        else:
            active_samples = self.samples
        
        # Sort samples by length for bucket grouping
        sample_lengths = [len(s[0]) for s in active_samples]
        sorted_indices = sorted(range(len(sample_lengths)), key=lambda i: sample_lengths[i])
        
        # Divide into buckets
        bucket_size = max(1, len(sorted_indices) // num_buckets)
        buckets = []
        for i in range(0, len(sorted_indices), bucket_size):
            bucket = sorted_indices[i:i + bucket_size]
            if len(bucket) >= batch_size:  # Only keep usable buckets
                buckets.append(bucket)
        
        class BucketDataset(Dataset):
            def __init__(self, samples, indices):
                self.samples = samples
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                real_idx = self.indices[idx]
                input_ids, labels = self.samples[real_idx]
                return (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                )
        
        def collate_fn(batch):
            input_ids = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # With bucket sampler, sequences have similar lengths → minimal padding
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            return {
                "input_ids": input_ids_padded,
                "labels": labels_padded,
            }
        
        # Create one DataLoader per bucket and chain them (simple approach)
        # Or use a flat dataset with indices grouped by length for each epoch
        all_indices = []
        for bucket in buckets:
            if shuffle:
                random.shuffle(bucket)
            all_indices.extend(bucket)
        
        # Reshuffle bucket order each epoch via a BatchSampler that groups by length
        class BucketBatchSampler(Sampler):
            def __init__(self, indices, lengths, batch_size, shuffle=True):
                self.indices = indices
                self.lengths = [lengths[i] for i in indices]
                self.batch_size = batch_size
                self.shuffle = shuffle
            
            def __iter__(self):
                # Sort by length, then create batches from consecutive items
                paired = list(zip(self.indices, self.lengths))
                paired.sort(key=lambda x: x[1])
                if self.shuffle:
                    # Shuffle within groups of batch_size * 10 to add randomness
                    # while keeping length similarity
                    chunks = [paired[i:i + self.batch_size * 10] 
                             for i in range(0, len(paired), self.batch_size * 10)]
                    random.shuffle(chunks)
                    paired = []
                    for chunk in chunks:
                        random.shuffle(chunk)
                        paired.extend(chunk)
                
                # Create batches from consecutive similar-length items
                for i in range(0, len(paired), self.batch_size):
                    batch = [p[0] for p in paired[i:i + self.batch_size]]
                    if len(batch) == self.batch_size:
                        yield batch
            
            def __len__(self):
                return len(self.indices) // self.batch_size
        
        batch_sampler = BucketBatchSampler(
            list(range(len(active_samples))),
            sample_lengths,
            batch_size,
            shuffle=shuffle,
        )
        
        return DataLoader(
            BucketDataset(active_samples, list(range(len(active_samples)))),
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    
    def get_stats(self) -> Dict:
        """Return dataset statistics."""
        if not self.samples:
            return {"num_samples": 0}
        
        lengths = [len(s[0]) for s in self.samples]
        return {
            "num_samples": len(self.samples),
            "min_len": min(lengths),
            "max_len": max(lengths),
            "avg_len": sum(lengths) / len(lengths),
            "total_tokens": sum(lengths),
        }
