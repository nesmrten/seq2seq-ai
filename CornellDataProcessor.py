import os
import re
import unicodedata
import numpy as np
from typing import Dict, List, Tuple
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from config import Config

class CornellDataProcessor:
    """
    A class for processing Cornell Movie Dialogs data.
    """

    def __init__(self, data_dir: str, min_length: int, max_length: int, validation_split: float,
                 train_file: str, val_file: str):
        self.data_dir = data_dir
        self.min_length = min_length
        self.max_length = max_length
        self.validation_split = validation_split
        self.train_file = train_file
        self.val_file = val_file

    def extract_lines(self, lines: Dict[str, str]) -> Dict[str, str]:
        """
        Extracts lines from the input dictionary.

        Args:
            lines (dict): A dictionary containing information about each line.

        Returns:
            dict: A dictionary containing only the relevant information for each line.
        """
        extracted_lines = {}
        for line_id, line_info in lines.items():
            if line_info['character_id'] != 'u':
                extracted_lines[line_id] = line_info['text']
        return extracted_lines

    def extract_conversations(self, lines: Dict[str, str], conversations: List[List[str]]) -> List[List[str]]:
        """
        Extracts conversations from the input list of conversations.

        Args:
            lines (dict): A dictionary containing information about each line.
            conversations (list): A list of conversations in the format [['line_id_1', 'line_id_2', ...], ...].

        Returns:
            list: A list of conversations in the format [['line_1', 'line_2', ...], ...].
        """
        extracted_conversations = []
        for conversation in conversations:
            # Get the text for each line in the conversation
            lines_in_conversation = [lines[line_id] for line_id in conversation]
            # Add the conversation to the list of extracted conversations
            extracted_conversations.append(lines_in_conversation)
        return extracted_conversations

    def load_lines(self, file_path: str, encoding: str = 'ISO-8859-1') -> Dict[str, str]:
        """Load the lines from the given file and return as a dictionary.

        Args:
            file_path (str): The path of the file to load.
            encoding (str, optional): The encoding of the file. Defaults to 'ISO-8859-1'.

        Returns:
            dict: A dictionary containing the lines with their IDs as keys.
        """
        lines = {}
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                line_parts = line.strip().split(' +++$+++ ')
                if len(line_parts) == 5:
                    lines[line_parts[0]] = line_parts[4]
        return lines

    def load_conversations(self, file_path: str, lines: Dict[str, str]) -> List[List[str]]:
        """Load the conversations from the given file and return as a list of lists of line IDs.

        Args:
            file_path (str): The path of the file to load.
            lines (dict): A dictionary containing the lines with their IDs as keys.

        Returns:
            list: A list of lists containing the line IDs for each conversation.
        """
        conversations = []
        with open(file_path, 'r', encoding='ISO-8859-1', errors='ignore') as f:
            for line in f:
                conversation_parts = line.strip().split(' +++$+++ ')[-1][1:-1].replace("'", "").split(", ")
                conversation_parts = [p for p in conversation_parts if p in lines]
                if len(conversation_parts) > 1:
                    conversations.append(conversation_parts)
        return conversations

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing unwanted characters and converting to lowercase.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        # Convert to lowercase
        text = text.lower()

        # Remove unwanted characters
        text = re.sub(r"([.!?])", r" \1", text)
        text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)

        return text

    def preprocess_sentence(self, sentence: str) -> List[str]:
        """
        Preprocesses a sentence by cleaning it and tokenizing it into a list of words.

        Args:
            sentence (str): The sentence to preprocess.

        Returns:
            List[str]: A list of preprocessed words.
        """
        # Clean the sentence
        sentence = self.clean_text(sentence)

        # Tokenize the sentence
        tokenizer = get_tokenizer('basic_english')
        tokens = tokenizer(sentence.strip())

        return tokens

    def preprocess_pairs(self, pairs: List[Tuple[str, str]]) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Preprocesses pairs of questions and answers.

        Args:
            pairs (List[Tuple[str, str]]): A list of pairs of questions and answers.

        Returns:
            Tuple[List[List[str]], List[List[str]]]: A tuple of preprocessed questions and preprocessed answers.
        """
        # Preprocess the questions and answers separately
        questions = []
        answers = []
        for pair in pairs:
            question = self.preprocess_sentence(pair[0])
            answer = self.preprocess_sentence(pair[1])
            if len(question) > 0 and len(answer) > 0:
                questions.append(question)
                answers.append(answer)

        return questions, answers

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
        word_freq = {}
        for word in all_words:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

        # Sort the words by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

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

    def generate_pairs(self, conversations: List[List[str]]) -> List[Tuple[str, str]]:
        """
        Generate question-answer pairs from the input conversations.

        Args:
            conversations (List[List[str]]): A list of conversations.

        Returns:
            List[Tuple[str, str]]: A list of question-answer pairs.
        """
        # Get pairs of conversational lines
        conversation_pairs = self.get_conversation_pairs(conversations, self.lines)

        # Filter pairs based on their length
        filtered_pairs = []
        for pair in conversation_pairs:
            if self.min_length <= len(pair[0].split()) <= self.max_length and \
                    self.min_length <= len(pair[1].split()) <= self.max_length:
                filtered_pairs.append(pair)

        return filtered_pairs

    def split_dataset(self, pairs: List[Tuple[str, str]], validation_split: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        Split the input dataset into training and validation sets.

        Args:
            pairs (List[Tuple[str, str]]): A list of question-answer pairs.
            validation_split (float): The ratio of validation set size to total dataset size.

        Returns:
            Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: A tuple of training and validation sets of question-answer pairs.
        """
        # Shuffle the dataset
        np.random.shuffle(pairs)

        # Split the dataset into training and validation sets
        num_validation_samples = int(len(pairs) * validation_split)
        train_pairs = pairs[:-num_validation_samples]
        val_pairs = pairs[-num_validation_samples:]

        return train_pairs, val_pairs

    def write_to_file(self, pairs: List[Tuple[str, str]], file_path: str) -> None:
        """
        Write the input pairs to file.

        Args:
            pairs (List[Tuple[str, str]]): A list of question-answer pairs.
            file_path (str): The path of the file to write to.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(pair[0] + '\t' + pair[1] + '\n')

    def build_dataset(self) -> None:
        """
        Build the dataset from the Cornell Movie Dialogs corpus.
        """
        # Load lines from file
        self.lines = self.load_lines(os.path.join(self.data_dir, 'movie_lines.txt'))

        # Load conversations from file
        self.conversations = self.load_conversations(os.path.join(self.data_dir, 'movie_conversations.txt'), self.lines)

        # Generate pairs of conversational lines
        pairs = self.generate_pairs(self.conversations)

        # Split dataset into training and validation sets
        train_pairs, val_pairs = self.split_dataset(pairs, self.validation_split)

        # Write pairs to file
        self.write_to_file(train_pairs, self.train_file)
        self.write_to_file(val_pairs, self.val_file)
        
    if __name__ == "__main__":
        data_processor = CornellDataProcessor(DATA_DIR)
        data_processor.build_dataset()