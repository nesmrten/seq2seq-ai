import os
import random
from typing import List, Tuple
from collections import Counter
import json


class CornellDataProcessor:
    def __init__(self, data_dir='data/cornell movie-dialogs corpus', min_word_frequency=2):
        self.data_dir = data_dir
        self.lines_file = os.path.join(data_dir, 'movie_lines.txt')
        self.conversations_file = os.path.join(data_dir, 'movie_conversations.txt')
        self.metadata_file = os.path.join(data_dir, 'movie_characters_metadata.txt')
        self.vocab_file = os.path.join(data_dir, 'vocab.txt')
        self.train_file = os.path.join(data_dir, 'train.txt')
        self.val_file = os.path.join(data_dir, 'val.txt')
        self.max_length = 10
        self.min_word_frequency = min_word_frequency
        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

    def load_lines(self):
        with open(self.lines_file, 'r', encoding='iso-8859-1') as f:
            lines = f.readlines()
        return lines

    def load_conversations(self):
        with open(self.conversations_file, 'r', encoding='iso-8859-1') as f:
            conversations = f.readlines()
        return conversations

    def load_character_metadata(self):
        with open(self.metadata_file, 'r', encoding='iso-8859-1') as f:
            character_metadata = f.readlines()
        return character_metadata

    def extract_conversation_ids(self, line):
        line_parts = line.split(' +++$+++ ')
        conversation_ids = line_parts[-1].strip()[1:-1].replace("'", "").split(', ')
        return conversation_ids

    def extract_conversations(self, conversations):
        conversations_dict = {}
        id2name = {}
        for conversation in conversations:
            conversation_ids = self.extract_conversation_ids(conversation)
            for id in conversation_ids:
                if id not in conversations_dict:
                    line = self.id2line.get(id)
                    if line:
                        conversations_dict[id] = line
                        id2name[line.split(' ')[0]] = line.split(' ')[-1]

        return conversations_dict, id2name

    def extract_pairs(self):
        conversations = self.load_conversations()
        self.id2line = dict()
        for line in self.load_lines():
            parts = line.split(' +++$+++ ')
            if len(parts) == 5:
                self.id2line[parts[0]] = parts[4]

        conversation_pairs = []
        for conversation in conversations:
            conversation_ids = self.extract_conversation_ids(conversation)
            for i in range(len(conversation_ids) - 1):
                input_line_id = conversation_ids[i]
                target_line_id = conversation_ids[i+1]
                input_line = self.id2line.get(input_line_id)
                target_line = self.id2line.get(target_line_id)
                if input_line and target_line:
                    conversation_pairs.append([input_line, target_line])
        return conversation_pairs

    def build_vocabulary(self, pairs):
        questions = []
        answers = []
        for pair in pairs:
            questions.append(pair[0])
            answers.append(pair[1])

        word_counter = Counter()
        for sentence in questions + answers:
            for word in sentence.split():
                word_counter[word] += 1

        vocab_list = sorted(
            [word for word, count in word_counter.items() if count >= self.min_word_frequency])

        # Add special tokens to the vocabulary
        vocab_list = self.special_tokens + vocab_list

        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            for word in vocab_list:
                f.write(word + '\n')

        print('Vocabulary size:', len(vocab_list))
        print('Saved vocabulary to', self.vocab_file)
        
    def build_word2idx(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            words = f.read().splitlines()
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.idx2word = {idx: word for idx, word in enumerate(words)}

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
    
    def preprocess_pairs(self, pairs: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Preprocesses the pairs of questions and answers.

        Args:
            pairs (List[Tuple[str, str]]): A list of pairs of questions and answers.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: The preprocessed pairs of questions and answers.
        """
        questions = []
        answers = []
        for pair in pairs:
            question = self.tokenize(pair[0])
            answer = self.tokenize(pair[1])
            if len(question) <= self.max_length and len(answer) <= self.max_length:
                questions.append(question)
                answers.append(answer)

        return questions, answers

    def save_data(self, pairs: List[Tuple[str, str]], train_frac: float = 0.8, shuffle: bool = True, train_file: str = 'train.txt', val_file: str = 'val.txt') -> None:
            """
            Saves the preprocessed data to disk.

            Args:
                pairs (List[Tuple[str, str]]): A list of pairs of questions and answers.
                train_frac (float): The fraction of the data to use for training.
                shuffle (bool): Whether to shuffle the data before splitting into training and validation sets.
                train_file (str): The filename to use for the training data.
                val_file (str): The filename to use for the validation data.
            """
            # Preprocess the pairs
            questions, answers = self.preprocess_pairs(pairs)

            # Split the data into training and validation sets
            indices = list(range(len(questions)))
            if shuffle:
                random.shuffle(indices)
            split_idx = int(len(indices) * train_frac)
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            # Save the training and validation data to disk
            with open(train_file, 'w', encoding='utf-8', newline='\n') as f:
                for i in train_indices:
                    question = questions[i]
                    answer = answers[i]
                    f.write('\t'.join(question).replace('\n', '') + '\t' +
                            '\t'.join(answer).replace('\n', '') + '\n')

            with open(val_file, 'w', encoding='utf-8', newline='\n') as f:
                for i in val_indices:
                    question = questions[i]
                    answer = answers[i]
                    f.write('\t'.join(question).replace('\n', '') + '\t' +
                            '\t'.join(answer).replace('\n', '') + '\n')


if __name__ == '__main__':
    # Define the data directory and file paths
    DATA_DIR = 'data/cornell movie-dialogs corpus/'
    vocab_file = os.path.join(DATA_DIR, 'vocab.json')
    train_file = os.path.join(DATA_DIR, 'train.txt')
    val_file = os.path.join(DATA_DIR, 'val.txt')

    # Initialize the data processor
    data_processor = CornellDataProcessor(data_dir=DATA_DIR)

    # Extract conversation pairs
    pairs = data_processor.extract_pairs()

    # Build the vocabulary
    data_processor.build_vocabulary(pairs)

    # Build the word-to-index mapping
    data_processor.build_word2idx()

    # Save the preprocessed data
    data_processor.save_data(pairs, train_file=train_file, val_file=val_file)

    # Print vocabulary size
    print(f'The vocabulary has {len(data_processor.word2idx)} words.')

    # Get the cutoff for the training set
    train_cutoff = int(0.8 * len(pairs))

    # Print the number of training pairs
    print(f'The training data has {train_cutoff} pairs.')



