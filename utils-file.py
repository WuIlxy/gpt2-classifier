import torch
import torch.nn as nn
import os
import tempfile
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import HfApi

def compute_loss(logprobs, targets, attention_mask):
    """
    Task 3: compute cross-entropy loss for next-word prediction
    
    Args:
        logprobs: Output log probabilities from the model (B, S, V)
        targets: Target token IDs (B, S)
        attention_mask: Attention mask (B, S)
    
    Returns:
        Average loss
    """
    # Shift targets for next-word prediction
    shifted_logprobs = logprobs[:, :-1, :].contiguous()
    shifted_targets = targets[:, 1:].contiguous()
    shifted_mask = attention_mask[:, 1:].contiguous()

    # Compute cross entropy loss
    loss_fct = nn.NLLLoss(ignore_index=0)  # Assuming 0 is the pad token ID

    # Flatten the tensors
    shifted_logprobs = shifted_logprobs.view(-1, shifted_logprobs.size(-1))
    shifted_targets = shifted_targets.view(-1)

    # Apply mask and compute loss
    active_loss = shifted_mask.view(-1) == 1
    active_logprobs = shifted_logprobs[active_loss]
    active_targets = shifted_targets[active_loss]

    return loss_fct(active_logprobs, active_targets)

def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Generate text from the model given a prompt
    
    Args:
        model: The trained model
        tokenizer: Tokenizer instance
        prompt: Text prompt to start generation
        max_length: Maximum length of generated text
    
    Returns:
        Generated text
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)

    # Get the end of sequence token id
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, attention_mask)
            next_token_logprobs = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logprobs, dim=-1).unsqueeze(-1)

            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=-1)

    return tokenizer.decode(input_ids[0])

def save_to_hub(model, model_name, token):
    """
    Save model to Hugging Face Hub
    
    Args:
        model: Model to save
        model_name: Repository name on Hugging Face Hub
        token: Authentication token
    """
    # Initialize Hugging Face API
    api = HfApi()

    # First create the repository
    try:
        api.create_repo(repo_id=model_name, token=token, exist_ok=True)
        print(f"Repository {model_name} created or already exists")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    # Create a temporary directory to save the model
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save model
        torch.save(model.state_dict(), os.path.join(tmp_dir, "pytorch_model.bin"))

        # Upload to Hub
        try:
            api.upload_file(
                path_or_fileobj=os.path.join(tmp_dir, "pytorch_model.bin"),
                path_in_repo="pytorch_model.bin",
                repo_id=model_name,
                token=token
            )
            print(f"Model saved to Hugging Face Hub: {model_name}")
            print(f"You can find your model at: https://huggingface.co/{model_name}")
        except Exception as e:
            print(f"Error uploading file: {e}")

def get_latest_checkpoint(checkpoint_dir):
    """
    Get the latest checkpoint from a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    
    Returns:
        Epoch number of the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None
    # Extract epoch numbers and find the latest
    epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
    return max(epochs) if epochs else None

def process_age_column(age_str):
    """
    Process age string to get minimum age
    
    Args:
        age_str: Age string from the dataset
    
    Returns:
        Minimum age as integer
    """
    age_str = str(age_str).strip()
    if '+' in age_str:
        return int(age_str.replace('+', ''))
    if '-' in age_str:
        return int(age_str.split('-')[0])
    try:
        return int(age_str)
    except ValueError:
        return None

class BookDataset(Dataset):
    """
    Dataset for book descriptions and ages
    """
    def __init__(self, descriptions, ages, tokenizer, max_length=512):
        self.descriptions = descriptions
        self.ages = ages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        desc = str(self.descriptions[idx])
        age = self.ages[idx]
        # Adjust age to be zero-indexed (1-20 -> 0-19)
        adjusted_age = max(0, min(age - 1, 19))

        # Add [cls] token at end for classification
        desc = desc + " [cls]"

        encoding = self.tokenizer(
            desc,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(adjusted_age, dtype=torch.long)
        }
