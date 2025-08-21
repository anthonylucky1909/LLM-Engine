# 🚀 LLM-Engine: Build, Train, and Deploy Large Language Models for Chatbots  

<!-- Project Badges -->
<p align="center">
  <!-- GitHub Release -->
  <a href="https://github.com/anthonyhuang1909/LLM-Engine/releases">
    <img src="https://img.shields.io/github/v/release/anthonyhuang1909/LLM-Engine?style=flat-square&logo=github" alt="Latest Release" />
  </a>
  <a href="https://github.com/anthonyhuang1909/LLM-Engine">
    <img src="https://img.shields.io/github/downloads/anthonyhuang1909/LLM-Engine/total?style=flat-square" alt="GitHub downloads" />
  </a>

  <!-- Python -->
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python versions" />
  </a>

  <!-- License -->
  <a href="https://github.com/anthonyhuang1909/LLM-Engine/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/anthonyhuang1909/LLM-Engine?style=flat-square" alt="License" />
  </a>

  <!-- Issues -->
  <a href="https://github.com/anthonyhuang1909/LLM-Engine/issues">
    <img src="https://img.shields.io/github/issues/anthonyhuang1909/LLM-Engine?style=flat-square&logo=github" alt="Open issues" />
  </a>
  <!-- Coverage -->
  <a href="https://app.codecov.io/gh/anthonyhuang1909/LLM-Engine">
    <img src="https://codecov.io/gh/anthonyhuang1909/LLM-Engine/branch/main/graph/badge.svg?style=flat-square" alt="Coverage" />
  </a>
</p>

---

## 📖 Overview  

**LLM-Engine** is a modular platform to **build, train, evaluate, and deploy large language models (LLMs)** for chatbot applications.  
It implements a **GPT-2 style Transformer decoder**, providing efficient natural language understanding and generation with customizable architecture.  

---

## 🧩 GPT-2 Model Architecture  

The GPT-2 model follows the **Transformer decoder architecture**, consisting of stacked layers of:  
- Multi-head self-attention  
- Position-wise feed-forward layers  
- Residual connections & layer normalization  

This design enables capturing **long-range dependencies** and **contextual information** effectively.  

![GPT-2 Model Architecture](diagram/diagram.jpeg)  

*Reference:* Vaswani et al. (2017). *Attention is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  

---

## ⚡ Quick Start  

### Using Docker  
```bash
chmod +x run.sh
./run.sh
```  

### Manual Setup  
```bash
pip install -r requirements.txt
python3 inference.py
```  

---

## 📂 Dataset Preparation  

For experiments, you can use open-source datasets such as:  

- [OpenAssistant Conversations](https://huggingface.co/datasets/OpenAssistant/oasst1)  
- [Dialog datasets on Hugging Face](https://huggingface.co/datasets)  

Example:  
```bash
python scripts/prepare_dataset.py --input raw_dataset.jsonl --output data/word_level_dataset.csv
```  

---

## 🏋️ Training the Model  

```bash
python3 train.py   --epochs 10   --lr 0.0001   --d_model 512   --n_layers 8   --n_heads 8   --dropout 0.1   --save_path Model.pth   --print_samples 3   --tie_embeddings
```  

**Arguments:**  
- `--epochs` : Training epochs  
- `--lr` : Learning rate  
- `--d_model` : Embedding dimension  
- `--n_layers` : Transformer decoder layers  
- `--n_heads` : Attention heads  
- `--dropout` : Dropout rate  
- `--save_path` : Save model path  
- `--print_samples` : Print training samples  
- `--tie_embeddings` : Tie input/output embeddings  

---

## 📦 Pretrained Model  

You can load pretrained weights from Hugging Face:  

```bash
git clone https://huggingface.co/anthonyhuang1909/LLM-Engine
```  

Includes:  
- `Model.pth` – pretrained weights  
- `vocab.json` – tokenizer vocabulary  

---

## ⚠️ Disclaimer  

This project is intended for **educational & research purposes**.  
It demonstrates the principles of Transformer-based models on a smaller scale.  

## 📜 License  

Released under the **MIT License**.  

---

*Last updated: 2025-08-21*  
