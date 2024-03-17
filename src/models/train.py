import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from src.data.data_set import ShakespeareDataset


def train_model(train_txt, valid_txt, best_model_path="best_shakespeare_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.requires_grad_(True)

    train_dataset = ShakespeareDataset(train_txt, tokenizer)
    valid_dataset = ShakespeareDataset(valid_txt, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    num_epochs = int(input("Number of epochs: "))
    learning_rate = 1e-4
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_loss = 1000

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch.to(device)
            labels = input_ids.clone()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {average_loss:.4f}")

        model.eval()
        total_loss_valid = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch.to(device)
                labels = input_ids.clone()
                outputs = model(input_ids, labels=labels)
                total_loss_valid += outputs.loss.item()

        valid_loss = total_loss_valid / len(valid_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {valid_loss:.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
