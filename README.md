# LLM-Engine: A Platform to Build and Deploy Large Language Models for Chatbots

---

## Overview

**LLM-Engine** is a comprehensive platform designed to facilitate the building, training, evaluation, and deployment of large language models (LLMs) specifically tailored for chatbot applications. This project implements a GPT-2 style model using the Transformer architecture, enabling efficient natural language understanding and generation.

---

## GPT-2 Model Architecture

The GPT-2 model is based on the Transformer decoder architecture, which consists of *N* stacked Transformer decoder blocks. Each block includes multi-head self-attention mechanisms and position-wise feed-forward layers. This architecture enables the model to capture long-range dependencies and contextual information effectively, making it highly suitable for language modeling tasks.

The image below illustrates the overall GPT-2 architecture:

![GPT-2 Model Architecture](diagram/diagram.jpeg)

*Source:* Yang, Steve; Ali, Zulfikhar; Wong, Bryan (2023). *FLUID-GPT (Fast Learning to Understand and Investigate Dynamics with a Generative Pre-Trained Transformer): Efficient Predictions of Particle Trajectories and Erosion*.  
DOI: [10.26434/chemrxiv-2023-ppk9s](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/64e3304660d17b562a9db0f4/original/fluid-gpt-fast-learning-to-understand-and-investigate-dynamics-with-a-generative-pre-trained-transformer-efficient-predictions-of-particle-trajectories-and-erosion.pdf)

---

## Quick Start

### Using Docker

To build and run the inference environment using Docker:

```bash
chmod +x run.sh
./run.sh
```

This script will build the Docker image and start the inference container.

### Manual Setup

If you prefer to run the project manually, install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run the inference script:

```bash
python3 inference.py
```

---

## Dataset Preparation

This project uses conversational datasets suitable for chatbot training.

To download the latest dataset version, use the KaggleHub Python package:

```python
import kagglehub

# Download latest dataset
path = kagglehub.dataset_download("noahpersaud/89k-chatgpt-conversations")

print("Path to dataset files:", path)
```

Prepare the dataset for training by running:

```bash
python scripts/prepare_dataset.py --input chatlogs.jsonl --output data/word_level_dataset.csv
```

---

## Training the Model

To train the model from scratch, use the training script with configurable parameters:

```bash
python3 train.py \
    --epochs 10 \
    --lr 0.0001 \
    --d_model 512 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --save_path Model.pth \
    --print_samples 3 \
    --tie_embeddings
```

### Training Arguments Description:

- `--epochs` : Number of training epochs (default: 10)  
- `--lr` : Learning rate (default: 1e-4)  
- `--d_model` : Dimensionality of model embeddings (default: 512)  
- `--n_layers` : Number of Transformer decoder layers (default: 8)  
- `--n_heads` : Number of attention heads (default: 8)  
- `--dropout` : Dropout rate (default: 0.1)  
- `--save_path` : Path to save the trained model (default: "Model.pth")  
- `--print_samples` : Number of sample outputs printed during training (default: 3)  
- `--tie_embeddings` : Use tied input and output embeddings (flag)

---

## Pretrained Model and Vocabulary

You may download a pretrained model and vocabulary files for quick experimentation:

```bash
git clone https://huggingface.co/anthonyhuang1909/LLM-Engine
```

The repository contains:

- `Model.pth` — pretrained model weights  
- `vocab.json` — vocabulary mapping for tokenization

---
## Disclaimer

This project implements a small-scale GPT model designed for educational and experimental purposes. It demonstrates the core principles of Transformer-based language models while maintaining simplicity for easier understanding and modification.

---
## Citation

If you use this project or model in your research, please cite the original GPT-2 paper:

```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019},
  institution={OpenAI}
}
```

---

## License

This project is released under the MIT License.

---

For questions or contributions, please open an issue or submit a pull request.

---

*Last updated: 2025-08-10*
