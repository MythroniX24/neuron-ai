"""
Continuum SLM — Main Entry Point.

Usage:
    # Chat UI (demo mode)
    python run.py chat --demo

    # Chat UI with model
    python run.py chat --model checkpoints/best_model.pt

    # Train the model
    python run.py train --data data/corpus.txt --epochs 10

    # Tokenizer training
    python run.py tokenize --train data/corpus.txt --save tokenizer.json
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_chat(args):
    """Launch the chat UI."""
    from continuum.ui.app import app
    print(f"Starting Continuum Chat on http://{args.host}:{args.port}")
    print("Open this URL on your phone to chat!")
    if args.demo:
        print("Running in DEMO mode (no model loaded)")
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_train(args):
    """Train the model."""
    import torch
    from continuum.model.model import create_continuum_nano
    from continuum.training.trainer import ContinuumTrainer
    from continuum.tokenizer.bpe import ContinuumTokenizer

    print("=" * 60)
    print("Continuum SLM Training")
    print("=" * 60)

    # Load or create tokenizer
    if args.tokenizer and os.path.exists(args.tokenizer):
        print(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = ContinuumTokenizer.load(args.tokenizer)
    else:
        print("Creating new tokenizer...")
        tokenizer = ContinuumTokenizer(vocab_size=8000)
        if args.data and os.path.exists(args.data):
            print(f"Training tokenizer on {args.data}")
            with open(args.data, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
            tokenizer.train(texts, verbose=True)
            if args.tokenizer:
                tokenizer.save(args.tokenizer)

    # Create model
    print("Creating Continuum-Nano model (~5M params)...")
    model = create_continuum_nano()
    print(f"Model: {model.num_params:,} parameters")

    # Create trainer
    trainer = ContinuumTrainer(
        model=model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    # Load data
    print(f"Loading training data from {args.data}")
    # Simple text dataset (user should prepare proper dataset)
    with open(args.data, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    # Tokenize
    print("Tokenizing dataset...")
    encoded = []
    for text in texts:
        tokens = tokenizer.encode_with_special(text, add_bos=True, add_eos=True)
        encoded.append(torch.tensor(tokens))

    # Create DataLoader (simplified)
    from torch.utils.data import DataLoader, TensorDataset
    # Pad sequences
    from torch.nn.utils.rnn import pad_sequence
    padded = pad_sequence(encoded, batch_first=True, padding_value=0)
    labels = padded.clone()
    dataset = TensorDataset(padded, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Dummy val loader (split from train)
    val_size = max(1, len(dataset) // 10)
    train_dataset = TensorDataset(padded[:-val_size], labels[:-val_size])
    val_dataset = TensorDataset(padded[-val_size:], labels[-val_size:])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")
    print("Starting training...")

    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        seq_len_start=args.seq_len_start,
        seq_len_end=args.seq_len_end,
    )

    print("Training complete!")
    trainer.save_checkpoint(args.epochs, is_best=True)


def cmd_tokenize(args):
    """Train or use the tokenizer."""
    from continuum.tokenizer.bpe import ContinuumTokenizer

    if args.train:
        print(f"Training tokenizer on {args.train}...")
        with open(args.train, "r") as f:
            texts = [line.strip() for line in f if line.strip()]

        tokenizer = ContinuumTokenizer(vocab_size=args.vocab_size)
        tokenizer.train(texts, verbose=True)

        save_path = args.save or "tokenizer.json"
        tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path} (vocab: {tokenizer.vocab_size_actual})")

    elif args.encode:
        tokenizer = ContinuumTokenizer.load(args.load or "tokenizer.json")
        tokens = tokenizer.encode(args.encode)
        print(f"Encoded: {tokens}")
        print(f"Decoded: {tokenizer.decode(tokens)}")

    elif args.decode:
        tokenizer = ContinuumTokenizer.load(args.load or "tokenizer.json")
        ids = [int(x) for x in args.decode.split(",")]
        print(f"Decoded: {tokenizer.decode(ids)}")


def main():
    parser = argparse.ArgumentParser(description="Continuum SLM")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Launch chat UI")
    chat_parser.add_argument("--host", default="0.0.0.0")
    chat_parser.add_argument("--port", type=int, default=5000)
    chat_parser.add_argument("--demo", action="store_true", help="Demo mode (no model)")
    chat_parser.add_argument("--debug", action="store_true")
    chat_parser.add_argument("--model", help="Model checkpoint path")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Training data file")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.1)
    train_parser.add_argument("--seq_len_start", type=int, default=32)
    train_parser.add_argument("--seq_len_end", type=int, default=512)
    train_parser.add_argument("--tokenizer", help="Tokenizer path (save/load)")
    train_parser.add_argument("--checkpoint_dir", default="checkpoints")
    train_parser.add_argument("--device", default="cpu")

    # Tokenizer command
    tok_parser = subparsers.add_parser("tokenize", help="Tokenizer operations")
    tok_parser.add_argument("--train", help="Corpus to train tokenizer on")
    tok_parser.add_argument("--vocab_size", type=int, default=8000)
    tok_parser.add_argument("--save", help="Save path")
    tok_parser.add_argument("--load", help="Load path")
    tok_parser.add_argument("--encode", help="Text to encode")
    tok_parser.add_argument("--decode", help="Comma-separated IDs to decode")

    args = parser.parse_args()

    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
