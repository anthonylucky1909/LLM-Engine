# train.py (place at project root)
import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

# import from your project structure
from data.vocab import build_vocab, save_vocab, encode_dataset
from data.dataset import GPTDataset, split_dataset
from model.gpt import GPTModel, GPTConfig
from training.trainer import train  

# config values (you can also read from config.py if you prefer)
INPUT_CSV = "dataset/word_level_dataset.csv"      # adjust if your file name differs
VOCAB_PATH = "dataset/vocab.json"
INV_VOCAB_PATH = "dataset/inv_vocab.json"
ENCODED_CSV = "dataset/encoded_dataset.csv"
TRAIN_CSV = "dataset/train_encoded.csv"
VAL_CSV = "dataset/val_encoded.csv"

MAX_SEQ_LEN = 64
BATCH_SIZE = 32
VAL_RATIO = 0.1


def main(args):
    # 1) Build vocab if missing
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(INV_VOCAB_PATH):
        print("Building vocabulary...")
        vocab = build_vocab(INPUT_CSV)
        save_vocab(vocab, VOCAB_PATH, INV_VOCAB_PATH)
        print(f"Saved vocab (size={len(vocab)}) to {VOCAB_PATH}")
    else:
        print("Loading existing vocab...")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded vocab (size={len(vocab)})")

    # 2) Encode dataset if missing
    if not os.path.exists(ENCODED_CSV):
        print("Encoding dataset...")
        encode_dataset(INPUT_CSV, ENCODED_CSV, vocab)
        print("Encoded dataset saved to", ENCODED_CSV)
    else:
        print("Encoded dataset already exists:", ENCODED_CSV)

    # 3) Split into train/val if missing
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV)):
        print("Splitting dataset into train/val...")
        split_dataset(ENCODED_CSV, TRAIN_CSV, VAL_CSV, val_ratio=VAL_RATIO)
    else:
        print("Train/val CSVs already present.")

    # 4) Build DataLoaders
    print("Loading datasets...")
    train_dataset = GPTDataset(TRAIN_CSV, vocab, MAX_SEQ_LEN)
    val_dataset = GPTDataset(VAL_CSV, vocab, MAX_SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5) Create model config and model
    cfg = GPTConfig(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=MAX_SEQ_LEN,
        dropout=args.dropout,
        tie_word_embeddings=getattr(args, "tie_embeddings", False),
    )
    model = GPTModel(cfg)

    # 6) Prepare inv_vocab mapping for printing during training
    inv_vocab = {str(idx): word for word, idx in vocab.items()}

    # 7) Call training loop implemented in training/trainer.py
    print("Starting training...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        inv_vocab=inv_vocab,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        pad_token_id=vocab["<PAD>"],
        print_samples=args.print_samples,
    )

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--save_path", type=str, default="Model.pth")
    p.add_argument("--print_samples", type=int, default=3)
    p.add_argument("--tie_embeddings", action="store_true")
    args = p.parse_args()

    main(args)
