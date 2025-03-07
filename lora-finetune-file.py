import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

from model import AgeClassificationModelWithLoRA
from utils import process_age_column, BookDataset

def train_age_classifier(model, train_loader, test_loader, num_epochs=5, device='cuda'):
    """
    Task 5: Train the age classifier using LoRA
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimize LoRA and classification head parameters with specified learning rate
    optimizer = optim.AdamW([
        {'params': model.lora_layers.parameters(), 'lr': 5e-4},
        {'params': model.classification_head.parameters(), 'lr': 5e-4}
    ])

    # Cosine Annealing with 4000 warm-up steps
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4000, eta_min=1e-5)

    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}:')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {val_loss/len(test_loader):.4f}')
        print(f'Accuracy: {accuracy:.2f}%')
        print(f'Best Accuracy: {best_accuracy:.2f}%\n')

    return best_accuracy

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune GPT2 for age classification with LoRA')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--data_path', type=str, default='children_books.csv', help='Path to children books CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained model on Hugging Face Hub')
    parser.add_argument('--hub_token', type=str, required=True, help='Authentication token for Hugging Face Hub')
    parser.add_argument('--lora_rank', type=int, default=8, help='Rank for LoRA layers')
    parser.add_argument('--lora_alpha', type=int, default=16, help='Alpha scaling for LoRA layers')
    args = parser.parse_args()

    # Initialize model with token for loading
    model = AgeClassificationModelWithLoRA(
        model_path=args.model_path,
        token=args.hub_token,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    # Load and process data
    print("Loading and processing children's books dataset...")
    df = pd.read_csv(args.data_path)
    df['min_reading_age'] = df['Reading_age'].apply(process_age_column)
    df = df.dropna(subset=['min_reading_age', 'Desc'])
    df['min_reading_age'] = df['min_reading_age'].astype(int)

    # Shuffle the dataframe first
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Take exactly 2000 samples for training, and the rest for testing
    train_df = df.iloc[:2000].reset_index(drop=True)
    test_df = df.iloc[2000:].reset_index(drop=True)

    # Verify the dataframes
    print("Total dataset size:", len(df))
    print("Train dataset size:", len(train_df))
    print("Test dataset size:", len(test_df))

    # Ensure the dataframes are not empty
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Dataset splitting resulted in empty DataFrames")

    # Create datasets
    train_dataset = BookDataset(train_df['Desc'], train_df['min_reading_age'], model.tokenizer)
    test_dataset = BookDataset(test_df['Desc'], test_df['min_reading_age'], model.tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_accuracy = train_age_classifier(
        model, 
        train_loader, 
        test_loader, 
        num_epochs=args.epochs, 
        device=device
    )
    
    print(f"Final best accuracy: {final_accuracy:.2f}%")
    print("Model saved as 'best_model.pth'")

if __name__ == "__main__":
    main()
