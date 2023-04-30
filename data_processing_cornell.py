import os
import json
import random
import argparse
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import configparser

# Load configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Access configuration values
data_dir = config.get('Data', 'DATA_DIR')
min_length = config.getint('Data', 'MIN_LENGTH')
max_length = config.getint('Data', 'MAX_LENGTH')


def load_conversations(conversations_path: str) -> List[List[str]]:
    """
    Load conversations from the given file and return as a list of lists of lines.

    Args:
        conversations_path (str): The path of the file containing the conversations.

    Returns:
        List[List[str]]: A list of lists containing the lines for each conversation.
    """
    conversations = []
    with open(conversations_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            conversation_parts = line.strip().split(' +++$+++ ')[-1][1:-1].replace("'", "").split(", ")
            conversation = [part.strip() for part in conversation_parts]
            conversations.append(conversation)
    return conversations


def load_lines(lines_path: str) -> dict:
    """
    Load lines from the given file and return as a dictionary.

    Args:
        lines_path (str): The path of the file containing the lines.

    Returns:
        dict: A dictionary containing the lines with their IDs as keys.
    """
    lines = {}
    with open(lines_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            line_parts = line.strip().split(' +++$+++ ')
            if len(line_parts) == 5:
                lines[line_parts[0]] = line_parts[4]
    return lines


def build_dataset(conversations: List[List[str]], lines: dict, data_dir: str, test_size: float = 0.1):
    """
    Build the dataset from the conversations and lines.

    Args:
        conversations (List[List[str]]): A list of lists containing the lines for each conversation.
        lines (dict): A dictionary containing the lines with their IDs as keys.
        data_dir (str): The directory to save the dataset files.
        test_size (float, optional): The fraction of the dataset to use as the test set. Defaults to 0.1.
    """
    # Split the dataset into train and test sets
    train_conversations, test_conversations = train_test_split(conversations, test_size=test_size, random_state=42)

    # Write the train set to file
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for conversation in train_conversations:
            for i in range(len(conversation) - 1):
                f.write(lines[conversation[i]] + '\t' + lines[conversation[i + 1]] + '\n')

    # Write the test set to file
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for conversation in test_conversations:
            for i in range(len(conversation) - 1):
                f.write(lines[conversation[i]] + '\t' + lines[conversation[i + 1]] + '\n')


def main(args):
    # Create the data directory if it doesn't exist
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # Load the conversations and lines
    conversations = load_conversations(args.conversations_path)
    lines = load_lines(args.lines_path)

    # Build the dataset
    build_dataset(conversations, lines, args.data_dir, args.test_size)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process Cornell Movie Dialogs dataset')
