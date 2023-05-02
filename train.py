import argparse
from models.seq2seq import Seq2Seq
from models.dataset import ChatbotDataset
from models.utils.tokenizer import Tokenizer
from config import Config
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from models.dataset import pad_collate_fn

#Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Define the argument parser
parser = argparse.ArgumentParser(description="Train a Seq2Seq model for a chatbot.")
parser.add_argument("data_dir", type=str, help="the directory containing the data")
parser.add_argument("vocab_file", type=str, help="the vocabulary file")
parser.add_argument("model_path", type=str, help="the path to save the trained model")

#Parse the command-line arguments
args = parser.parse_args()

#Ensure that the vocabulary file exists
vocab_file = args.vocab_file
if not os.path.exists(vocab_file):
    vocab_file = Config.VOCAB_FILE
if not os.path.exists(vocab_file):
    raise ValueError(f"Vocabulary file '{args.vocab_file}' does not exist.")

#Initialize the dataset and dataloader
dataset = ChatbotDataset(args.data_dir, vocab_file, min_word_freq=Config.MIN_WORD_FREQ, min_length=Config.MIN_LENGTH, max_length=Config.MAX_LENGTH)

dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=0))

#Initialize the model
model = Seq2Seq(dataset.tokenizer.get_vocab_size(), Config.EMBEDDING_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.DROPOUT).to(device)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

#Initialize the Tensorboard writer
writer = SummaryWriter(log_dir=Config.LOGS_DIR)

#Train the model
print("Training the model...")
for epoch in range(Config.NUM_EPOCHS):
    start_time = time.time()
    total_loss = 0

for i, batch in enumerate(dataloader):
    # Move the input and target data to the device
    inputs, inputs_lengths, targets, targets_lengths = batch
    inputs = inputs.to(device)
    targets = targets.to(device)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(inputs, inputs_lengths, targets)

    # Compute the loss
    loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))

    # Backward pass
    loss.backward()

    # Clip the gradients
    nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

    # Update the parameters
    optimizer.step()

    # Update the total loss
    total_loss += loss.item()

    # Print the loss every N batches
    if (i + 1) % Config.PRINT_EVERY == 0:
        avg_loss = total_loss / Config.PRINT_EVERY
        elapsed_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{Config.NUM_EPOCHS}], Batch [{i + 1}/{len(dataloader)}], Avg. Loss = {avg_loss:.4f}, "
              f"Elapsed Time = {elapsed_time:.2f}s")
        total_loss = 0
        start_time = time.time()

    # Log the training loss to Tensorboard
    writer.add_scalar("Training loss", loss.item(), epoch * len(dataloader) + i)

    #Print the average training loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f} time: {time.time()-start_time:.4f}s")
    
    #Save the model
    model_path = args.model_path
    if not model_path.endswith(".pt"):
        model_path += ".pt"
        torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

    #Close the Tensorboard writer
    writer.close()