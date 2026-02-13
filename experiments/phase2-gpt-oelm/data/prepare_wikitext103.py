#!/usr/bin/env python3
"""Prepare WikiText-103 dataset for language model training."""

import os
import numpy as np
from datasets import load_dataset
import tiktoken

print("Preparing WikiText-103 data...")
enc = tiktoken.get_encoding("gpt2")

# Load WikiText-103
print("Loading dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

# Process training set
print("Processing training set...")
train_text = "".join(dataset["train"]["text"])
train_tokens = enc.encode(train_text)
train_arr = np.array(train_tokens, dtype=np.uint16)

# Process validation set
print("Processing validation set...")
val_text = "".join(dataset["validation"]["text"])
val_tokens = enc.encode(val_text)
val_arr = np.array(val_tokens, dtype=np.uint16)

# Create directory and save
os.makedirs("data/wikitext103", exist_ok=True)
train_arr.tofile("data/wikitext103/train.bin")
val_arr.tofile("data/wikitext103/val.bin")

print(f"Training set: {len(train_tokens)} tokens")
print(f"Validation set: {len(val_tokens)} tokens")
print("Done!")
