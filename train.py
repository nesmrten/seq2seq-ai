import argparse
from models.seq2seq import Seq2Seq
from models.dataset import ChatbotDataset
from models.utils.tokenizer import Tokenizer
from config import Config
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the argument parser
parser = argparse.ArgumentParser(description="Evaluate a Seq2Seq model for a chatbot.")
parser.add_argument("data_dir", type=str, help="the directory containing the data")
parser.add_argument("vocab_file", type=str, help="the vocabulary file")
parser.add_argument("model_path", type=str, help="the path to the trained model")
parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE, help="the batch size (default: %(default)s)")

# Parse the command-line arguments
args = parser.parse_args()

# Ensure that the vocabulary file exists
vocab_file = args.vocab_file
if not os.path.exists(vocab_file):
    vocab_file = Config.VOCAB_FILE
if not os.path.exists(vocab_file):
    raise ValueError(f"Vocabulary file '{args.vocab_file}' does not exist.")

# Initialize the dataset and dataloader
dataset = ChatbotDataset(os.path.join(args.data_dir, "cornell_movie-dialogs_corpus"), vocab_file, min_word_freq=Config.MIN_WORD_FREQ, min_length=Config.MIN_LENGTH, max_length=Config.MAX_LENGTH)

test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=0))

# Initialize the model
model = Seq2Seq(dataset.tokenizer.get_vocab_size(), Config.EMBEDDING_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.DROPOUT).to(device)

# Load the trained model parameters
model.load_state_dict(torch.load(args.model_path))

# Evaluate the model
model.eval()

total_loss = 0
total_tokens = 0

with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        # Move the input and target data to the device
        inputs, inputs_lengths, targets, targets_lengths = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(inputs, inputs_lengths, targets)

        # Compute the loss
        loss = torch.nn.functional.cross_entropy(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1), ignore_index=0, reduction="sum")

        # Update the total loss and total number of tokens
        total_loss += loss.item()
        total_tokens += torch.sum(targets_lengths).item()

# Calculate the average loss per token
average_loss_per_token = total_loss / total_tokens

print(f"Average loss per token: {average_loss_per_token:.4f}")
