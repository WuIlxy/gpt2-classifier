# Language Model Training and Fine-tuning Project

This project implements the training of a GPT2-small model from scratch on the TinyStories dataset and fine-tunes it using Low Rank Adapters (LoRA) for age classification on children's books.

## Project Structure

- `gpt2_training.py`: Code for training the GPT2 model on TinyStories
- `lora_finetune.py`: Code for fine-tuning using LoRA on the children's books dataset
- `model.py`: Model definitions for both tasks
- `utils.py`: Utility functions for data processing and evaluation
- `requirements.txt`: Required packages
- `ESE3060Project.ipynb`: Jupyter notebook containing the full implementation

## Tasks Implemented

### Task 1: Custom GPT2 Model
Loaded the GPT2-small model with random initialization and modified the last layer to output log probability vectors.

### Task 2: Learning Rate Scheduler
Implemented a Cosine Annealing Scheduler with the AdamW optimizer, using 4000 warming steps and a learning rate of 5e-4.

### Task 3: Cross-Entropy Loss
Implemented cross-entropy loss for next-word prediction with proper masking for padding tokens.

### Task 4: Training on TinyStories
Trained the model on the TinyStories dataset and implemented a text generation function.

### Task 5: Age Classification with LoRA
Fine-tuned the model using Low Rank Adapters (LoRA) to classify the minimum reading age for children's books based on their descriptions.

## Results

### Text Generation Examples
After training on TinyStories, the model can generate coherent, simple stories based on prompts:

- Prompt: "On a rainy day"
  - Generated: "On a rainy day, Tom was playing in the garden. He saw a big, red ball. He wanted to play with it..."

### Age Classification
Using LoRA fine-tuning on the children's books dataset, we achieved an accuracy of 80.61% on the test set.

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Training GPT2 on TinyStories
```bash
python gpt2_training.py
```

### Fine-tuning with LoRA for Age Classification
```bash
python lora_finetune.py
```

## Dataset Sources
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- [Highly Rated Children Books And Stories](https://www.kaggle.com/datasets/thomaskonstantin/highly-rated-childrens-books-and-stories)

## References
1. TinyStories Dataset
2. TinyStories Tokenizer
3. LoRA: Low-Rank Adaptation of Large Language Models
4. Highly Rated Children Books And Stories Dataset
5. GPT2-small Model Specifications
