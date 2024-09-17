import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from MyDataset import TextDataset
from my_tokenizer import tokenizer
from model import Seq2SeqModel


if __name__ == "__main__":
    min_freq = 5
    max_length = 5100
    special_tokens = ["<unk>", "<pad>", "<eos>"]

    data = pd.read_pickle("mydata.pkl")
    #tokenizer = torchtext.data.utils.get_tokenizer("basic_english")


    # Create a dataset instance
    dataset = TextDataset(data[:120], tokenizer, max_length)
    del data
    n_words = dataset.get_vocab_len()
    print("Size of vocab:", n_words)
    batch_size = 12 # 32
    num_epochs = 5
    clip = 5

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Seq2SeqModel(vocab_size=n_words, embed_size=128, hidden_size=256, num_layers=2)
    model.load_state_dict(torch.load('GPT_model.pth', weights_only=True))

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.special_tokens["pad"])
    optimizer = Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            title = batch['title'].to(device)  # Shape: [batch_size, max_length]
            text = batch['text'].to(device)    # Shape: [batch_size, max_length]

            # Shift text by one for teacher forcing
            input_text = text[:, :-1]  # Remove the last token
            target_text = text[:, 1:]  # Remove the first token

            # Forward pass
            optimizer.zero_grad()
            outputs = model(title, input_text)  # Predict next token

            # Calculate loss
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), target_text.reshape(-1))
            loss.backward()

            # Gradient clipping (to prevent exploding gradients)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()

            # Free up GPU memory
            del title, text, input_text, target_text, outputs
            torch.cuda.empty_cache()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")



    torch.save(model.state_dict(), 'GPT_model.pth')
    #model.load_state_dict(torch.load('nn_model.pth'))