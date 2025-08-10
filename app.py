import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from collections import OrderedDict
from fastapi.responses import FileResponse
from model.gpt import GPTModel, GPTConfig

app = FastAPI()

# Load vocabularies and model once on startup
VOCAB_PATH = "dataset/vocab.json"
INV_VOCAB_PATH = "dataset/inv_vocab.json"
MODEL_CHECKPOINT = "model/Model.pth"

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
with open(INV_VOCAB_PATH, "r", encoding="utf-8") as f:
    inv_vocab = json.load(f)

cfg = GPTConfig(
    vocab_size=len(vocab),
    d_model=512,
    n_layers=8,
    n_heads=8,
    max_seq_len=20,
    dropout=0.1,
)

model = GPTModel(cfg)
state_dict = torch.load(MODEL_CHECKPOINT, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith("module.") else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def encode_input(text, vocab, max_seq_len):
    tokens = [vocab["<BOS>"]] + [vocab.get(w, vocab["<UNK>"]) for w in text.split()] + [vocab["<EOS>"]]
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    else:
        tokens += [vocab["<PAD>"]] * (max_seq_len - len(tokens))
    return torch.tensor([tokens], dtype=torch.long).to(device)


def decode_output(token_ids, inv_vocab):
    words = []
    for idx in token_ids:
        word = inv_vocab.get(str(idx.item()), "<UNK>")
        if word == "<EOS>":
            break
        if word not in ("<PAD>", "<BOS>"):
            words.append(word)
    return " ".join(words)


class Prompt(BaseModel):
    text: str

#############################################################################################################################
# BACKEND 
#############################################################################################################################
@app.get("/")
def index():
    return FileResponse("front-end/index.html")


@app.post("/generate")
def generate_text(prompt: Prompt):
    input_ids = encode_input(prompt.text, vocab, cfg.max_seq_len)
    eos_token_id = vocab.get("<EOS>")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=1.0,
            top_k=10,
            eos_token_id=eos_token_id,
        )
    output_ids = generated_ids[0].cpu()
    output_text = decode_output(output_ids, inv_vocab)
    return {"input": prompt.text, "output": output_text}
