import os

# Flask config
DEBUG = os.environ.get('DEBUG', True)
SECRET_KEY = os.environ.get('SECRET_KEY', 'Stargatesg-1!#$')
HOST = os.environ.get('HOST', 'localhost')
PORT = os.environ.get('PORT', 5000)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VOCAB_DIR = os.path.join(MODEL_DIR, 'vocab')
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pth')
TOKENIZER_FILE = os.path.join(VOCAB_DIR, 'tokenizer.json')

class Config:
    # Training parameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    teacher_forcing_ratio = 0.5

    # Model parameters
    hidden_size = 256
    embedding_size = 128
    num_layers = 2
    dropout = 0.5

    # Data parameters
    data_path = 'data/conversations.txt'
    max_length = 10

    # Inference parameters
    max_length_inference = 20

