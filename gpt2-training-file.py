import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import os
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from tqdm.auto import tqdm
import argparse

from model import CustomGPT2
from utils import compute_loss, generate_text, get_latest_checkpoint, save_to_hub

def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

def train_model(model, train_dataloader, num_epochs=10, checkpoint_dir='checkpoints', resume_from=None):
    """
    Task 4: Train the model on tiny-stores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=4000,  # Warming steps
        eta_min=1e-5
    )

    # Initialize start epoch
    start_epoch = 0

    # Load checkpoint if resuming training
    if resume_from is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{resume_from}.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            logprobs = model(input_ids, attention_mask)
            loss = compute_loss(logprobs, input_ids, attention_mask)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            current_train_loss = total_train_loss / (pbar.n + 1)

            pbar.set_postfix({
                'train_loss': f'{current_train_loss:.4f}'
            })

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': current_train_loss,
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")

        pbar.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train GPT2 on TinyStories dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=50000, help='Number of samples to use from TinyStories')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--context_size', type=int, default=512, help='Context size for tokenizing')
    parser.add_argument('--save_to_hub', action='store_true', help='Save model to Hugging Face Hub')
    parser.add_argument('--hub_model_name', type=str, default='gpt2-tinystories', help='Model name for Hugging Face Hub')
    parser.add_argument('--hub_token', type=str, default='', help='Authentication token for Hugging Face Hub')
    args = parser.parse_args()

    # Load dataset
    print("Loading TinyStories dataset...")
    tinystories_dataset = datasets.load_dataset("roneneldan/TinyStories")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset preparation
    NUM_TRAIN_SAMPLES = args.num_samples
    CONTEXT_SIZE = args.context_size

    if NUM_TRAIN_SAMPLES is not None:
        tinystories_dataset['train'] = tinystories_dataset['train'].select(range(NUM_TRAIN_SAMPLES))

    print("Tokenizing dataset...")
    tokenized_dataset = tinystories_dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=CONTEXT_SIZE
        ),
        batched=True
    )

    # Create dataloader
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Check for existing checkpoints
    print("Checking for existing checkpoints...")
    latest_epoch = get_latest_checkpoint(args.checkpoint_dir)

    if latest_epoch is None:
        print("Initializing new model with random weights...")
        model = CustomGPT2(tokenizer)
        resume_from = None
    else:
        print(f"Found checkpoint from epoch {latest_epoch}")
        model = CustomGPT2(tokenizer)
        resume_from = latest_epoch

    # Train model
    print("Starting/Resuming training...")
    train_model(model, train_dataloader, num_epochs=args.epochs, 
                checkpoint_dir=args.checkpoint_dir, resume_from=resume_from)

    # Generate example texts
    print("\nExample generated texts...")
    test_prompts = [
        "On a rainy day",
        "The big red truck",
        "In the kitchen",
        "Grace and her friend",
        "When school ended",
    ]

    print("\nGenerated Text Examples:")
    for prompt in test_prompts:
        print("\nPrompt:", prompt)
        generated = generate_text(model, tokenizer, prompt)
        print("Generated:", generated)
        print("-" * 50)

    # Save to Hugging Face Hub if requested
    if args.save_to_hub and args.hub_token:
        save_to_hub(model, args.hub_model_name, args.hub_token)

if __name__ == "__main__":
    main()
