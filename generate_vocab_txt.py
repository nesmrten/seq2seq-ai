import json
from models.utils.tokenizer import Tokenizer
from config import Config

# Load the dataset
tokenizer = Tokenizer()
tokenizer.load_from_file(Config.VOCAB_FILE)

# Save the dataset
with open("vocab.txt", "w") as f:
    for word in tokenizer.vocab:
        f.write(f"{word}\n")
