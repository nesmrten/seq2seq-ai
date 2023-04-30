import os

# Directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data/cornell movie-dialogs corpus/')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# Data processing
MIN_WORD_FREQ = 5
MIN_LENGTH = 5
MAX_LENGTH = 20
VALIDATION_SPLIT = 0.1

# Model hyperparameters
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 20
