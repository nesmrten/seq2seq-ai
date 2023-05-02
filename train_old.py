import os
import torch
import numpy as np
from models.seq2seq import Seq2Seq
from models.dataset import ChatbotDataset
from models.utils.tokenizer import Tokenizer
from torch.utils.data import DataLoader


DATA_DIR = 'data'
MODEL_DIR = 'models'
VOCAB_FILE = os.path.join(DATA_DIR, 'vocab.txt')
MODEL_FILE = os.path.join(MODEL_DIR, 'seq2seq.pt')
DATA_FILE = os.path.join(DATA_DIR, 'cornell movie-dialogs corpus/movie_lines.txt')
MIN_WORD_FREQ = 5
MIN_LENGTH = 5
MAX_LENGTH = 20
BATCH_SIZE = 32
EMBEDDING_SIZE = 256
HIDDEN_SIZE = 256
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
VALIDATION_SPLIT = 0.2


if __name__ == '__main__':
    # Load the vocabulary
    tokenizer = Tokenizer(VOCAB_FILE)

    # Load the dataset and split into training and validation datasets
    dataset = ChatbotDataset(DATA_FILE, VOCAB_FILE, MIN_WORD_FREQ, MIN_LENGTH, MAX_LENGTH)
    train_dataset, dev_dataset = dataset.split(split_ratio=VALIDATION_SPLIT)

    # Create data loaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.pad_collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dev_dataset.pad)




# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a sequence-to-sequence model for a chatbot.')
parser.add_argument('--data_dir', type=str, default='data/cornell movie-dialogs corpus/', help='Directory containing the data files.')
parser.add_argument('--models_dir', type=str, default='models/', help='Directory to save the trained models.')
parser.add_argument('--logs_dir', type=str, default='logs/', help='Directory to save the TensorBoard logs.')
parser.add_argument('--min_word_freq', type=int, default=5, help='Minimum frequency of a word to be included in the vocabulary.')
parser.add_argument('--min_length', type=int, default=5, help='Minimum length of a sentence to be included in the dataset.')
parser.add_argument('--max_length', type=int, default=20, help='Maximum length of a sentence to be included in the dataset.')
parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of the data to use for validation.')
parser.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in the RNN.')
parser.add_argument('--embedding_size', type=int, default=128, help='Size of the word embeddings.')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the RNN.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train.')
parser.add_argument('--save_every', type=int, default=5, help='Save the model every N epochs.')
parser.add_argument('--resume', type=str, default=None, help='Path to a saved model to resume training from.')
parser.add_argument('--run_name', type=str, default='default', help='Name of the current run for TensorBoard logging.')
args = parser.parse_args()

# Directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = 'data'
MODELS_DIR = os.path.join(PROJECT_DIR, args.models_dir)
LOGS_DIR = os.path.join(PROJECT_DIR, args.logs_dir)

# Data processing
MIN_WORD_FREQ = args.min_word_freq
MIN_LENGTH = args.min_length
MAX_LENGTH = args.max_length
VALIDATION_SPLIT = args.validation_split

# Load the tokenizer
tokenizer_path = os.path.join(PROJECT_DIR, 'models/tokenizer.json')
with open(tokenizer_path, 'r') as f:
    tokenizer = Tokenizer.from_json(json.load(f))

#Model hyperparameters
HIDDEN_SIZE = args.hidden_size
EMBEDDING_SIZE = args.embedding_size
NUM_LAYERS = args.num_layers
DROPOUT = args.dropout
LEARNING_RATE = args.learning_rate
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
SAVE_EVERY = args.save_every
RESUME_PATH = args.resume
RUN_NAME = args.run_name

#Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Create the directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

#Load the dataset
dataset = ChatbotDataset(os.path.join(DATA_DIR, 'cornell movie-dialogs corpus/movie_lines.txt'), tokenizer, MIN_WORD_FREQ, MIN_LENGTH, MAX_LENGTH)

#Split the dataset into training and validation sets
train_dataset, dev_dataset = dataset.split(VALIDATION_SPLIT)

#Create the DataLoader for the training set
train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)

#Create the DataLoader for the validation set
dev_data_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

    #Initialize the model
if RESUME_PATH is not None:
    # Load the saved model from disk
    model = Seq2Seq.load_from_file(RESUME_PATH)
else:
# Create a new model
    model = Seq2Seq(
input_size=tokenizer.vocab_size,
output_size=tokenizer.vocab_size,
hidden_size=HIDDEN_SIZE,
embedding_size=EMBEDDING_SIZE,
num_layers=NUM_LAYERS,
dropout=DROPOUT,
)

#Move the model to the device
model = model.to(device)

#Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#Initialize the criterion
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx['<pad>'])

#Create the TensorBoard SummaryWriter object
writer = SummaryWriter(os.path.join(LOGS_DIR, RUN_NAME))

#Train the model
train_loss, dev_loss, dev_acc = train_loop(model, train_data_loader, dev_data_loader, optimizer, criterion, device, NUM_EPOCHS, writer, MODELS_DIR, SAVE_EVERY)

#Close the TensorBoard SummaryWriter object
writer.close()

print(f'Training complete. Train Loss: {train_loss:.3f} - Dev Loss: {dev_loss:.3f} - Dev Acc: {dev_acc:.3f}')