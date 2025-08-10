import json
import torch
from collections import OrderedDict
from model.gpt import GPTModel, GPTConfig


def encode_input(text, vocab, max_seq_len):
    """
    Convert input text to token IDs tensor with padding.
    Adds <BOS> and <EOS> tokens.
    """
    tokens = [vocab["<BOS>"]] + [vocab.get(w, vocab["<UNK>"]) for w in text.split()] + [vocab["<EOS>"]]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    else:
        tokens += [vocab["<PAD>"]] * (max_seq_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.long)


def decode_output(token_ids, inv_vocab):
    """
    Decode token IDs back into human-readable text.
    Stops at <EOS> and skips <PAD> and <BOS>.
    """
    words = []
    for idx in token_ids:
        word = inv_vocab.get(str(idx.item()), "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<BOS>"):
            words.append(word)
    return " ".join(words)


def infer(
    model,
    input_text,
    vocab,
    inv_vocab,
    max_seq_len,
    max_new_tokens=20,
    temperature=1.0,
    top_k=10,
    device=None,
):
    """
    Generate text autoregressively from the model given an input prompt.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    input_ids = encode_input(input_text, vocab, max_seq_len).to(device)
    eos_token_id = vocab.get("<EOS>")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )

    output_ids = generated_ids[0].cpu()
    return decode_output(output_ids, inv_vocab)


def load_vocab(vocab_path, inv_vocab_path):
    """
    Load vocabulary and inverse vocabulary from JSON files.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(inv_vocab_path, "r", encoding="utf-8") as f:
        inv_vocab = json.load(f)
    return vocab, inv_vocab


def load_model(checkpoint_path, config):
    """
    Load the GPT model from checkpoint with config.
    Handles DataParallel prefix if present.
    """
    model = GPTModel(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def main():
    VOCAB_PATH = "dataset/vocab.json"
    INV_VOCAB_PATH = "dataset/inv_vocab.json"
    MODEL_CHECKPOINT = "model/Model.pth"

    print("Loading vocabularies...")
    vocab, inv_vocab = load_vocab(VOCAB_PATH, INV_VOCAB_PATH)

    print("Preparing model config and loading model...")
    cfg = GPTConfig(
        vocab_size=len(vocab),
        d_model=512,
        n_layers=8,
        n_heads=8,
        max_seq_len=20,
        dropout=0.1,
    )
    model = load_model(MODEL_CHECKPOINT, cfg)

    print("Model loaded successfully!")
    print("Type your prompt and press Enter (type 'quit' or 'exit' to stop).")

    while True:
        input_text = input("Input : ").strip()
        if input_text.lower() in ["quit", "exit"]:
            print("Exiting.")
            break

        output_text = infer(model, input_text, vocab, inv_vocab, cfg.max_seq_len)
        print("Output:", output_text)
        print("-" * 40)


if __name__ == "__main__":
    main()
