# train.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import AdamW

def decode_batch(batch_ids, inv_vocab):
    """Convert a batch of token IDs back into text sequences."""
    decoded = []
    for seq in batch_ids.cpu().numpy():
        words = []
        for idx in seq:
            word = inv_vocab.get(str(idx), "<UNK>")
            if word == "<EOS>":
                break
            if word not in ["<PAD>", "<BOS>"]:
                words.append(word)
        if not words:
            words = ["[EMPTY]"]
        decoded.append(" ".join(words))
    return decoded

def train_epoch(model, dataloader, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0
    for input_ids, target_ids in tqdm(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)  # (B, seq_len, vocab_size)

        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = target_ids.view(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, pad_token_id):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, target_ids in dataloader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_token_id)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(
    model, train_loader, val_loader, inv_vocab,
    epochs=10, lr=4e-4, save_path="gpt_model.pth",
    pad_token_id=0, print_samples=3
):
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, pad_token_id)
        val_loss = evaluate(model, val_loader, device, pad_token_id)
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Print sample predictions
        model.eval()
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                logits = model(input_ids)
                preds = logits.argmax(dim=-1)

                inputs_text = decode_batch(input_ids, inv_vocab)
                targets_text = decode_batch(target_ids, inv_vocab)
                preds_text = decode_batch(preds, inv_vocab)

                print("\nSamples:")
                for i in range(min(print_samples, len(inputs_text))):
                    print(f"Input    : {inputs_text[i]}")
                    print(f"Expected : {targets_text[i]}")
                    print(f"Predicted: {preds_text[i]}")
                    print("---")
                break  # only from first batch

        # Save checkpoint
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}\n")
