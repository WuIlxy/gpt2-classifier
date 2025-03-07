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
Example texts...

Generated Text Examples: 

Prompt: On a rainy day
Generated: On a rainy day, Tom was playing in the garden. He saw a big, red ball. He wanted to play with it. He ran to the ball and tried to grab it. But the ball was too high for him. He tried to jump, but he could not. He felt sad and angry.

He saw a big leaf on the ground. He thought it was a good leaf. He picked it up and held it in his hands. He wanted to make it look like a leaf. He
--------------------------------------------------

Prompt: The big red truck
Generated: The big red truck was in the park. It was red and shiny. The truck driver was driving a big truck. The truck driver was driving the truck.

The truck driver drove the truck to the park. He saw a big truck. The truck was red and shiny. The driver stopped the truck. He looked at the truck.

The truck driver smiled. He liked the truck. The truck driver drove the truck to the park. He saw a big truck. The truck was red. The truck
--------------------------------------------------

Prompt: In the kitchen
Generated: In the kitchen, there was a little girl named Lily. She was three years old and loved to play outside. One day, she was playing with her toys when she heard a loud noise. It was her mommy's voice calling her from the kitchen.

"Lily, come here please!" her mommy said.

Lily ran to the kitchen and saw her mommy's voice. "Hi mommy, what's wrong?" she asked.

"I'm trying to make
--------------------------------------------------

Prompt: Grace and her friendWhen school ended
Generated: Grace and her friendWhen school ended, they were very excited. They had been playing together all day and were going to the same school. Grace was so excited to go to school and learn something new.

When they arrived, Grace saw that the school classroom was very big and had lots of books. She was so excited to learn something new.

Grace and her friend studied the classroom. They had so much fun learning and learning together. They were so happy and excited to learn something new.

Grace
--------------------------------------------------

### Age Classification Results
Epoch 1/5: 100%
 250/250 [00:42<00:00,  5.90it/s]
Epoch 1:
Training Loss: 1.5133
Validation Loss: 1.1662
Accuracy: 59.73%
Best Accuracy: 59.73%

Epoch 2/5: 100%
 250/250 [00:42<00:00,  5.92it/s]
Epoch 2:
Training Loss: 1.2747
Validation Loss: 1.0230
Accuracy: 57.84%
Best Accuracy: 59.73%

Epoch 3/5: 100%
 250/250 [00:42<00:00,  5.92it/s]
Epoch 3:
Training Loss: 1.1633
Validation Loss: 0.8294
Accuracy: 80.30%
Best Accuracy: 80.30%

Epoch 4/5: 100%
 250/250 [00:42<00:00,  5.91it/s]
Epoch 4:
Training Loss: 1.0686
Validation Loss: 0.8833
Accuracy: 78.64%
Best Accuracy: 80.30%

Epoch 5/5: 100%
 250/250 [00:42<00:00,  5.91it/s]
Epoch 5:
Training Loss: 1.0180
Validation Loss: 0.6707
Accuracy: 80.61%
Best Accuracy: 80.61%

80.61465721040189
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
