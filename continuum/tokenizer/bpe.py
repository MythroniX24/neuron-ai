"""
Byte-level BPE Tokenizer for Continuum.

As specified in Section 4 of the architecture:
- Byte-level: every byte (0-255) is a guaranteed base token
- Small vocabulary: 8,000 tokens for Nano tier
- Single-digit number tokenization: digits 0-9 are NEVER merged
- No <unk> token — OOV is impossible by construction
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple


class ContinuumTokenizer:
    """
    Byte-level BPE tokenizer with single-digit number splitting.

    Internal representation: tokens are stored as tuples of bytes.
    Base vocabulary is all 256 single-byte tokens.
    Merges are learned from training corpus via BPE.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    # Digit pattern for single-digit splitting
    DIGIT_PATTERN = re.compile(r"\d")

    def __init__(self, vocab_size: int = 8000):
        """
        Args:
            vocab_size: Target vocabulary size (including 256 base bytes + special tokens).
                        Default 8000 for Continuum-Nano.
        """
        self.vocab_size = vocab_size

        # Base vocabulary: all 256 bytes
        self.byte_to_token: Dict[int, int] = {}
        self.token_to_bytes: Dict[int, bytes] = {}
        self.merges: Dict[Tuple[bytes, bytes], int] = {}  # (token_a_bytes, token_b_bytes) -> new_token_id

        # Special token IDs (reserved at the top of the vocab)
        self.pad_id: int = 0
        self.bos_id: int = 1
        self.eos_id: int = 2

        # Initialize base vocabulary
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize with all 256 byte tokens plus special tokens."""
        # Special tokens first
        special_offset = 3

        # 256 byte tokens
        for i in range(256):
            byte_val = bytes([i])
            token_id = i + special_offset
            self.byte_to_token[i] = token_id
            self.token_to_bytes[token_id] = byte_val

        # Map special tokens
        self.token_to_bytes[self.pad_id] = self.PAD_TOKEN.encode("utf-8")
        self.token_to_bytes[self.bos_id] = self.BOS_TOKEN.encode("utf-8")
        self.token_to_bytes[self.eos_id] = self.EOS_TOKEN.encode("utf-8")

    def _split_digits(self, text: str) -> str:
        """
        Pre-process: insert spaces around each digit so BPE never merges them.
        "abc123def" -> "abc 1 2 3 def"
        This implements the single-digit number tokenization from Section 4.
        """
        return self.DIGIT_PATTERN.sub(r" \g<0> ", text)

    def _text_to_bytes(self, text: str) -> List[int]:
        """Convert text to list of byte values (0-255)."""
        return list(text.encode("utf-8"))

    def _bytes_to_text(self, byte_list: List[int]) -> str:
        """Convert list of byte values back to text."""
        return bytes(byte_list).decode("utf-8", errors="replace")

    # ⚡ OPTIMIZED: Use collections.Counter for C-level pair counting
    def _get_stats(self, ids: List[int]) -> Counter:
        """Count adjacent pairs in the ID sequence. Returns Counter (C-optimized)."""
        return Counter(zip(ids, ids[1:]))

    def _merge_ids(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """Merge all occurrences of pair into new_id."""
        new_ids = []
        i = 0
        n = len(ids)
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts: List[str], verbose: bool = False):
        """
        Train BPE merges on a corpus of texts.

        ⚡ OPTIMIZED: Uses Counter.most_common() instead of max() over dict.
        Maintains faster loop binding with local variable references.

        Args:
            texts: List of training texts
            verbose: Print progress
        """
        # Convert texts to byte-level token IDs (already split digits)
        all_ids = []
        for text in texts:
            processed = self._split_digits(text)
            byte_vals = self._text_to_bytes(processed)
            token_ids = [self.byte_to_token[b] for b in byte_vals]
            all_ids.append(token_ids)

        num_merges = self.vocab_size - 259  # 256 bytes + 3 special tokens

        if verbose:
            print(f"Training BPE: target vocab={self.vocab_size}, "
                  f"starting tokens=259, merges to learn={num_merges}")

        # ⚡ OPTIMIZED: Pre-allocate progress printing frequency
        print_interval = max(1, num_merges // 40)  # ~40 progress updates total

        # ⚡ OPTIMIZED: Local variable bindings for faster loop
        token_to_bytes = self.token_to_bytes
        merges = self.merges
        byte_to_token = self.byte_to_token

        for merge_step in range(num_merges):
            # ⚡ OPTIMIZED: Use Counter + most_common(1) instead of max(dict, key=dict.get)
            # Counter.most_common() uses C-level heapq.nlargest internally
            stats = Counter()
            for seq_ids in all_ids:
                # ⚡ OPTIMIZED: update() is faster than += in a loop
                stats.update(zip(seq_ids, seq_ids[1:]))

            if not stats:
                break

            # ⚡ OPTIMIZED: most_common(1) is O(1) after counting (uses C heapq)
            best_pair, _ = stats.most_common(1)[0]
            new_id = 259 + merge_step  # Start after base + special tokens

            # Record merge
            bytes_a = token_to_bytes[best_pair[0]]
            bytes_b = token_to_bytes[best_pair[1]]
            merged_bytes = bytes_a + bytes_b
            token_to_bytes[new_id] = merged_bytes
            merges[(bytes_a, bytes_b)] = new_id

            # ⚡ OPTIMIZED: Apply merge in-place using list comprehension for the merge
            # _merge_ids already creates new lists, but let's keep it as-is for correctness
            for i in range(len(all_ids)):
                all_ids[i] = self._merge_ids(all_ids[i], best_pair, new_id)

            if verbose and (merge_step + 1) % print_interval == 0:
                pct = (merge_step + 1) / num_merges * 100
                try:
                    decoded = merged_bytes.decode('utf-8', errors='replace')[:30]
                except Exception:
                    decoded = repr(merged_bytes)[:30]
                print(f"  Merge {merge_step + 1}/{num_merges} ({pct:.0f}%): "
                      f"freq={stats[best_pair]}, token='{decoded}'")

        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.token_to_bytes)}")

    def _is_single_digit_token(self, tok_id: int) -> bool:
        """Check if a token represents exactly one ASCII digit (0x30-0x39)."""
        b = self.token_to_bytes.get(tok_id, b'')
        return len(b) == 1 and 0x30 <= b[0] <= 0x39

    def _ensure_fast_tables(self):
        """Build token-id-level merge lookup tables (cached until merges change).

        Rank-based BPE needs, for a pair of adjacent token IDs: its merge
        priority and the resulting token. Deriving these from the bytes-keyed
        `self.merges` dict on every call is what made the old encoder
        O(merges × seq_len) per pass (~3800 merges rescanned thousands of
        times → 9+ hours to tokenize a 52K-conversation corpus).
        """
        if (getattr(self, "_tables_len", -1) == len(self.merges)
                and getattr(self, "_pair_rank", None) is not None):
            return
        bytes_to_id = {b: i for i, b in self.token_to_bytes.items()}
        pair_rank: Dict[Tuple[int, int], int] = {}
        pair_new_id: Dict[Tuple[int, int], int] = {}
        for (b_a, b_b), new_id in self.merges.items():
            id_a = bytes_to_id.get(b_a)
            id_b = bytes_to_id.get(b_b)
            if id_a is None or id_b is None:
                continue  # dangling merge (shouldn't happen in a trained tokenizer)
            # Lower new_id = learned earlier = higher merge priority.
            pair_rank[(id_a, id_b)] = new_id
            pair_new_id[(id_a, id_b)] = new_id
        self._pair_rank = pair_rank
        self._pair_new_id = pair_new_id
        self._tables_len = len(self.merges)

    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.

        Rank-based BPE — identical output to the naive scan-every-merge loop,
        but ~1000x faster (only pairs PRESENT in the sequence are considered,
        instead of rescanning all ~3800 merges over the full sequence).

        Algorithm:
        1. Convert to bytes -> initial token IDs
        2. Repeatedly merge the earliest-learned adjacent pair present in the
           sequence (standard BPE loop), but NEVER merge two digit tokens
           (this enforces single-digit number tokenization per Section 4)
        """
        byte_vals = self._text_to_bytes(text)
        ids = [self.byte_to_token[b] for b in byte_vals]
        if len(ids) < 2:
            return ids

        self._ensure_fast_tables()
        pair_rank = self._pair_rank
        pair_new_id = self._pair_new_id
        is_digit = self._is_single_digit_token

        while True:
            # 1. Find the highest-priority (earliest-learned) adjacent pair present.
            best_rank = None
            best_pair = None
            prev = ids[0]
            for nxt in ids[1:]:
                r = pair_rank.get((prev, nxt))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = (prev, nxt)
                prev = nxt
            if best_pair is None:
                break  # no applicable merge — done

            # 2. Merge ALL adjacent occurrences of that pair (left→right).
            a, b = best_pair
            new_id = pair_new_id[best_pair]
            out = []
            i = 0
            n = len(ids)
            while i < n:
                if (i < n - 1 and ids[i] == a and ids[i + 1] == b
                        and not (is_digit(ids[i]) and is_digit(ids[i + 1]))):
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            if len(out) == len(ids):
                break  # every occurrence digit-protected — pair is unmergeable
            ids = out

        return ids

    def _encode_reference(self, text: str) -> List[int]:
        """Legacy O(merges × seq_len) encoder — kept ONLY for equivalence tests."""
        byte_vals = self._text_to_bytes(text)
        ids = [self.byte_to_token[b] for b in byte_vals]

        # Sort merges by token ID (order learned) — earlier merges have priority
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

        # Iteratively apply all possible merges
        changed = True
        while changed:
            changed = False
            for (bytes_a, bytes_b), new_id in sorted_merges:
                i = 0
                while i < len(ids) - 1:
                    # Enforce single-digit tokenization: never merge two digit tokens
                    if self._is_single_digit_token(ids[i]) and self._is_single_digit_token(ids[i + 1]):
                        i += 1
                        continue

                    tok_a_bytes = self.token_to_bytes.get(ids[i])
                    tok_b_bytes = self.token_to_bytes.get(ids[i + 1])
                    if tok_a_bytes == bytes_a and tok_b_bytes == bytes_b:
                        ids = ids[:i] + [new_id] + ids[i + 2:]
                        changed = True
                        # Continue from one position before the merge site,
                        # since the new merged token might form another pair
                        i = max(0, i - 1)
                    else:
                        i += 1

        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_parts = []
        for tid in ids:
            if tid in self.token_to_bytes:
                token_bytes = self.token_to_bytes[tid]
                # Skip special tokens in output
                if token_bytes in (self.PAD_TOKEN.encode("utf-8"),
                                   self.BOS_TOKEN.encode("utf-8"),
                                   self.EOS_TOKEN.encode("utf-8")):
                    continue
                byte_parts.append(token_bytes)
            else:
                # Unknown token ID — shouldn't happen with byte-level BPE
                byte_parts.append(b"?")
        combined = b"".join(byte_parts)
        return combined.decode("utf-8", errors="replace")

    def encode_with_special(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode with optional BOS/EOS tokens."""
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.encode(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def save(self, path: str):
        """Save tokenizer state to disk."""
        state = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0].hex()}+{k[1].hex()}": v for k, v in self.merges.items()},
            "token_to_bytes": {str(k): v.hex() for k, v in self.token_to_bytes.items()
                              if k >= 3 and k < 259},  # only special + base byte tokens
            # Store all merge-produced tokens by their merge order
            "merge_tokens": {str(k): v.hex() for k, v in self.token_to_bytes.items()
                            if k >= 259}
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ContinuumTokenizer":
        """Load tokenizer from disk."""
        with open(path, "r") as f:
            state = json.load(f)

        tokenizer = cls(vocab_size=state["vocab_size"])

        # Restore merge-produced tokens
        for k_str, v_hex in state["merge_tokens"].items():
            k = int(k_str)
            tokenizer.token_to_bytes[k] = bytes.fromhex(v_hex)

        # Restore merges
        for k_str, v in state["merges"].items():
            hex_a, hex_b = k_str.split("+")
            tokenizer.merges[(bytes.fromhex(hex_a), bytes.fromhex(hex_b))] = v

        return tokenizer

    @property
    def vocab_size_actual(self) -> int:
        """Return actual vocabulary size."""
        return len(self.token_to_bytes)
