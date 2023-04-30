import argparse
import json
import os
import re
import string
import random


def preprocess_text(text):
    """
    Performs basic text preprocessing steps on the input text.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation marks
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove leading/trailing whitespaces
    text = text.strip()

    return text


def process_dialogues(dialogues):
    """
    Processes the dialogues in the input list of dialogues.

    Args:
        dialogues (list): A list of dialogues in the format [{'dialogue': [{'text': str, 'persona': list}, ...]}, ...].

    Returns:
        list: A list of processed dialogues in the format [{'dialogue': [{'text': str, 'persona': str}, ...]}, ...].
    """
    processed_dialogues = []
    for dialogue in dialogues:
        processed_dialogue = {'dialogue': []}

        # Combine all the persona strings into a single string
        persona_str = ' '.join([persona for turn in dialogue['dialogue'] for persona in turn['persona']])

        for turn in dialogue['dialogue']:
            processed_turn = {}
            processed_turn['text'] = preprocess_text(turn['text'])
            processed_turn['persona'] = preprocess_text(persona_str)
            processed_dialogue['dialogue'].append(processed_turn)

        processed_dialogues.append(processed_dialogue)

    return processed_dialogues


def split_train_dev_test(data, dev_size=0.1, test_size=0.1, random_seed=42):
    """
    Splits the input data into training, development, and test sets.

    Args:
        data (list): The input data to split.
        dev_size (float, optional): The size of the development set as a fraction of the total data size. Defaults to 0.1.
        test_size (float, optional): The size of the test set as a fraction of the total data size. Defaults to 0.1.
        random_seed (int, optional): The random seed to use for the shuffle. Defaults to 42.

    Returns:
        tuple: A tuple of the training, development, and test sets.
    """
    # Shuffle the data
    random.seed(random_seed)
    random.shuffle(data)

    # Calculate the sizes of the train, dev, and test sets
    data_size = len(data)
    dev_size = int(data_size * dev_size)
    test_size = int(data_size * test_size)
    train_size = data_size - dev_size - test_size

    # Split the data into train, dev, and test sets
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]

    return train_data, dev_data, test_data


def save_data(data, output_dir):
    with open(output_dir, 'w') as f:
        json.dump(data, f)
    print('Saved data to', output_dir)

if __name__ == '__main__':
    # Add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/personachat/train_self_original.json',
                        help='Path to the input JSON file')
    parser.add_argument('--output_dir', type=str, default='data/personachat/processed',
                        help='Path to the output directory')
    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load the data from the input file
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Process the dialogues in the data
    processed_data = process_dialogues(data)

    # Split the data into train, dev, and test sets
    train_data, dev_data, test_data = split_train_dev_test(processed_data)

    # Save the processed data
    save_data(train_data, os.path.join(args.output_dir, 'train.json'))
    save_data(dev_data, os.path.join(args.output_dir, 'dev.json'))
    save_data(test_data, os.path.join(args.output_dir, 'test.json'))
