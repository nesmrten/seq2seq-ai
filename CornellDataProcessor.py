import os
import json
import random
from typing import List, Tuple, Dict

import torch
from torch import Tensor


DATA_DIR = 'data/cornell movie-dialogs corpus/'

class CornellDataProcessor:
    def __init__(self, data_dir: str, min_word_freq: int = 5, max_len: int = 50):
        """
        Initialize the CornellDataProcessor object.

        Args:
            data_dir (str): The directory where the data files are stored.
            min_word_freq (int): The minimum frequency for a word to be included in the vocabulary.
            max_len (int): The maximum length of a sequence (in tokens).
        """
        self.data_dir = data_dir
        self.min_word_freq = min_word_freq
        self.max_len = max_len

        # Define the file paths
        self.metadata_file = os.path.join(data_dir, 'movie_characters_metadata.txt')
        self.conversations_file = os.path.join(data_dir, 'movie_conversations.txt')
        self.lines_file = os.path.join(data_dir, 'movie_lines.txt')
        self.vocab_file = os.path.join(data_dir, 'vocab.json')
        self.train_file = os.path.join(data_dir, 'train.txt')
        self.val_file = os.path.join(data_dir, 'val.txt')

        # Initialize the variables
        self.word_freq = {}
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        self.conversations = []
        self.questions = []
        self.answers = []

    def preprocess_pairs(self, pairs: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Preprocesses the input pairs of questions and answers by tokenizing, lowercasing, and adding the start and end
        tokens.

        Args:
            pairs (List[Tuple[str, str]]): A list of pairs of questions and answers.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: The preprocessed questions and answers as lists of lists of tokens.
        """
        # Tokenize and lowercase the questions and answers
        tokenized_pairs = [(self.tokenize(sentence.lower()), self.tokenize(response.lower())) for sentence, response in pairs]

        # Add the start and end tokens to the answer
        preprocessed_pairs = [(question, ['<sos>'] + answer + ['<eos>']) for question, answer in tokenized_pairs]

        # Filter out pairs with questions or answers that are too long
        preprocessed_pairs = [(question, answer) for question, answer in preprocessed_pairs if len(question) <= self.max_len and len(answer) <= self.max_len]

        # Separate the questions and answers
        questions, answers = zip(*preprocessed_pairs)

        return list(questions), list(answers)

    def build_vocabulary(self, pairs: List[Tuple[str, str]]) -> None:
        """
        Builds the vocabulary from the input pairs of questions and answers.

        Args:
            pairs (List[Tuple[str, str]]): A list of pairs of questions and answers.
        """
        # Preprocess the pairs
        questions, answers = self.preprocess_pairs(pairs)

        # Flatten the questions and answers into one list
        all_words = [word for sentence in questions + answers for word in sentence]

        # Count the frequency of each word
        for word in all_words:
            if word not in self.word_freq:
                self.word_freq[word] = 1
            else:
                self.word_freq[word] += 1

        # Sort the words by frequency
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Create the vocabulary
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        for i, (word, freq) in enumerate(sorted_words):
            if freq < self.min_word_freq:
                break
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

# Save the vocabulary to disk
with open(self.vocab_file, 'w', encoding='utf-8') as f:
    json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)

def load_vocab(self) -> None:
    """
    Loads the vocabulary from the vocabulary file.

    Raises:
        FileNotFoundError: If the vocabulary file does not exist.
    """
    if not os.path.exists(self.vocab_file):
        raise FileNotFoundError(f'Vocabulary file {self.vocab_file} not found')

    # Load the vocabulary from disk
    with open(self.vocab_file, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    self.word2idx = vocab['word2idx']
    self.idx2word = vocab['idx2word']

def load_data(self) -> None:
    """
    Loads the data from the data files.
    """
    # Load the vocabulary
    self.load_vocab()

    # Load the conversations
    self.load_conversations()

    # Split the conversations into questions and answers
    self.extract_pairs()

    # Build the vocabulary from the questions and answers
    self.build_vocabulary(list(zip(self.questions, self.answers)))

    # Save the training and validation data to disk
    self.save_data()

def load_conversations(self) -> None:
    """
    Loads the conversations from the conversations file.
    """
    # Load the conversations from the file
    with open(self.conversations_file, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()

    # Parse the conversations
    for line in lines:
        line = line.strip()
        if line:
            conversation = json.loads(line)
            conversation = [self.extract_text_from_line(line_id) for line_id in conversation['utterance_ids']]
            self.conversations.append(conversation)

def extract_text_from_line(self, line_id: str) -> str:
    """
    Extracts the text from a line in the lines file.

    Args:
        line_id (str): The ID of the line to extract.

    Returns:
        str: The text of the line.
    """
    # Find the line with the given ID
    with open(self.lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line = line.strip()
            if line_id in line:
                parts = line.split(' +++$+++ ')
                text = parts[-1]
                return text

    # If the line wasn't found, return an empty string
    return ''

def extract_pairs(self) -> None:
    """
    Extracts pairs of questions and answers from the conversations.
    """
    # Iterate through each conversation
    for conversation in self.conversations:
        # Iterate through each pair of adjacent utterances
        for i in range(len(conversation) - 1):
            self.questions.append(conversation[i])
            self.answers.append(conversation[i + 1])

def save_data(self) -> None:
    """
    Saves the training and validation data to disk.
    """
    # Split the data into training and validation sets
    indices = list(range(len(self.questions)))
    random.shuffle(indices)
    train_indices = indices[:int(len(indices) * 0.9)]
    val_indices = indices[int(len(indices) * 0.9):]

# Save the training data to disk
self.save_data(self.train_file, self.questions[:train_cutoff], self.answers[:train_cutoff])

# Save the validation data to disk
self.save_data(self.val_file, self.questions[train_cutoff:], self.answers[train_cutoff:])

def tokenize(self, sentence: str) -> List[str]:
    """
    Tokenizes a sentence.

    Args:
        sentence (str): The sentence to tokenize.

    Returns:
        List[str]: The tokenized sentence.
    """
    # Split the sentence into tokens
    tokens = sentence.split()

    # Remove any tokens that contain non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    return tokens

def save_data(self, file_path: str, questions: List[List[str]], answers: List[List[str]]) -> None:
    """
    Saves the questions and answers to disk.

    Args:
        file_path (str): The path to the file where the data will be saved.
        questions (List[List[str]]): A list of questions as lists of tokens.
        answers (List[List[str]]): A list of answers as lists of tokens.
    """
    # Combine the questions and answers into pairs
    pairs = list(zip(questions, answers))

    # Shuffle the pairs
    random.shuffle(pairs)

    # Open the file for writing
    with open(file_path, 'w', encoding='utf-8') as f:
        # Write each pair to the file as a JSON object
        for question, answer in pairs:
            f.write(json.dumps({'question': question, 'answer': answer}) + '\n')

if __name__ == '__main__':
    # Define the data directory and file paths
    DATA_DIR = 'data/cornell movie-dialogs corpus/'
    metadata_file = os.path.join(DATA_DIR, 'movie_characters_metadata.txt')
    conversations_file = os.path.join(DATA_DIR, 'movie_conversations.txt')
    lines_file = os.path.join(DATA_DIR, 'movie_lines.txt')
    vocab_file = os.path.join(DATA_DIR, 'vocab.json')
    train_file = os.path.join(DATA_DIR, 'train.txt')
    val_file = os.path.join(DATA_DIR, 'val.txt')

    # Initialize the data processor
    processor = CornellDataProcessor(data_dir=DATA_DIR)

    # Load the lines from the lines file
    lines = processor.load_lines()

    # Load the conversations from the conversations file
    conversations = processor.load_conversations(lines)

    # Load the metadata from the metadata file
    character_metadata = processor.load_character_metadata()

    # Create a dictionary mapping character IDs to character names
    id2name = processor.get_character_id2name(character_metadata)

    # Extract the pairs of questions and answers from the conversations
    pairs = processor.extract_pairs(conversations, id2name)

    # Build the vocabulary
    processor.build_vocabulary(pairs)

    # Split the pairs into training and validation sets
    train_cutoff = int(len(pairs) * 0.8)

    # Save the vocabulary to disk
    processor.save_vocab()

    print(f'The vocabulary has {len(processor.word2idx)} words.')

    # Save the training and validation data to disk
    processor.save_data(train_file, processor.questions[:train_cutoff], processor.answers[:train_cutoff])
    processor.save_data(val_file, processor.questions[train_cutoff:], processor.answers[train_cutoff:])

    print(f'The training data has {len(processor.questions[:train_cutoff])} pairs.')

