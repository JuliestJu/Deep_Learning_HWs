
!pip install sacremoses
import os
import random
from dataclasses import dataclass

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,  DataCollatorForSeq2Seq
from datasets import load_dataset
import sentencepiece
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

ds = load_dataset("Helsinki-NLP/europarl", "en-sk")

# Split Data
train_valid_split = ds['train'].train_test_split(test_size=0.2, seed=42)

train_data = train_valid_split['train']
valid_data = train_valid_split['test']

train_data = train_data.shuffle(seed=42).select(range(int(len(train_data) * 0.05)))
valid_data = valid_data.shuffle(seed=42).select(range(int(len(valid_data) * 0.05)))

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sk")

def tokenize_function(examples):
    source_texts = [item['en'] for item in examples['translation']]
    target_texts = [item['sk'] for item in examples['translation']]
    return tokenizer(
        source_texts, 
        text_target=target_texts, 
        truncation=True, 
        max_length=128
    )
    
train_tokenized = train_data.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["translation"]
)

valid_tokenized = valid_data.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["translation"]
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="Helsinki-NLP/opus-mt-en-sk")

train_dataloader = DataLoader(train_tokenized, shuffle=True, batch_size=16, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_tokenized, batch_size=16, collate_fn=data_collator)
print(len(train_dataloader))

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]

        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]

        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Use the top layer hidden state for attention
        attn_weights = self.attention(hidden[-1, :, :], encoder_outputs)  # [batch_size, src_len]
        attn_weights = attn_weights.unsqueeze(1)  # [batch_size, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hid_dim]

        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hid_dim]
        # Use hidden as is: shape [n_layers, batch_size, hid_dim]
        output, hidden = self.rnn(rnn_input, hidden)  # output: [batch_size, 1, hid_dim], hidden: [n_layers, batch_size, hid_dim]

        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        # prediction: [batch_size, output_dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(src)

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # Define the start token ID safely
        start_token_id = tokenizer.bos_token_id
        if start_token_id is None:
            start_token_id = tokenizer.pad_token_id

        # Initialize the first decoder input token
        input = torch.full((batch_size,), start_token_id, dtype=torch.long, device=self.device)
    
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
    
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t, :] = output
    
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
    
            # Get the target token at time-step t
            next_token = trg[:, t]
    
            # Replace any -100 indices (padding for labels) with the model's prediction (top1)
            next_token = torch.where(next_token == -100, top1, next_token)
    
            input = next_token if teacher_force else top1

        return outputs

INPUT_DIM = len(tokenizer.get_vocab())  # Vocabulary size for the source language
OUTPUT_DIM = len(tokenizer.get_vocab())  # Vocabulary size for the target language
EMB_DIM = 32
HID_DIM = 64
N_LAYERS = 2
DROPOUT = 0.7

attention = Attention(HID_DIM)
encoder = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
decoder = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, DROPOUT, attention)

model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

def train_one_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(dataloader):
        
        src = batch['input_ids'].to(device)       # Source tokens
        trg = batch['labels'].to(device)          # Target tokens

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass with dynamic teacher forcing ratio
        output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)  # Exclude <sos> token
        trg = trg[:, 1:].reshape(-1)  # Exclude <sos> token

        # Calculate loss
        loss = criterion(output, trg)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def train_model(model, train_dataloader, valid_dataloader, optimizer, criterion, device, n_epochs, initial_teacher_forcing_ratio=0.9, min_teacher_forcing_ratio=0.1):
    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/{n_epochs}')

        # Calculate teacher forcing ratio for this epoch
        teacher_forcing_ratio = max(
            initial_teacher_forcing_ratio - (epoch / n_epochs) * (initial_teacher_forcing_ratio - min_teacher_forcing_ratio),
            min_teacher_forcing_ratio
        )
        print(f"Teacher Forcing Ratio: {teacher_forcing_ratio:.2f}")
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Training phase
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, teacher_forcing_ratio)
        train_losses.append(train_loss)

        # Validation phase
        valid_loss = evaluate_model(model, valid_dataloader, criterion, device, teacher_forcing_ratio)
        valid_losses.append(valid_loss)

        print(f'Training Loss: {train_loss:.4f} | Validation Loss: {valid_loss:.4f}')

        # Step the scheduler to adjust the learning rate after each epoch
        # scheduler.step()
    
    return train_losses, valid_losses

def evaluate_model(model, dataloader, criterion, device, teacher_forcing_ratio=0):
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0

    with torch.no_grad():  # Disable gradient computation for evaluation
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            trg = batch['labels'].to(device)

            # Use the provided teacher_forcing_ratio in the forward pass
            output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)

            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # Exclude <sos> token
            trg = trg[:, 1:].reshape(-1)

            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

N_EPOCHS = 32

# Train the Model
train_losses, valid_losses = train_model(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    device=DEVICE,
    n_epochs=N_EPOCHS,
    initial_teacher_forcing_ratio=0.95,
    min_teacher_forcing_ratio=0.2
)

# Save Losses for Analysis
loss_data = {'train_loss': train_losses, 'valid_loss': valid_losses}
torch.save(loss_data, 'loss_data.pth')

def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

plot_losses(train_losses, valid_losses)

torch.save(model.state_dict(), 'seq2seq_model.pth')

# Save the tokenizer (important for decoding)
tokenizer.save_pretrained('./tokenizer')

print("Model and tokenizer saved successfully!")