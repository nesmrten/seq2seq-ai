import os
import re
import unicodedata
import numpy as np
from typing import Dict, List, Tuple
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class CornellDataProcessor:
    """
    A class for processing Cornell Movie Dialogs data.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.lines = self.load_lines(os.path.join(data_dir, 'movie_lines.txt'))
        self.conversations = self.load_conversations(os.path.join(data_dir, 'movie_conversations.txt'), self.lines)

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

    def load_lines(self, file_path: str) -> Dict[str, str]:
        """
        Load lines from the given file and returns a dictionary of lines.

        Args:
            file_path (str): The path to the file to load lines from.

        Returns:
            dict: A dictionary of lines.
        """
        lines = {}
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                if line.startswith('+++$+++'):
                    line_parts = line.strip().split('+++$+++')
                    if len(line_parts) == 5:
                        line_id = line_parts[0].strip()
                        line_text = line_parts[4].strip()
                        lines[line_id] = line_text

        return lines


def load_conversations(self, file_path: str, lines: Dict[str, str]) -> List[List[str]]:
    """
    Load conversations from the given file and returns a list of conversations, where each conversation is a list of line IDs.

    Args:
        file_path (str): The path to the file to load conversations from.
        lines (dict): A dictionary of lines to map the conversations to.

    Returns:
        list: A list of conversations, where each conversation is a list of line IDs.
    """
    conversations = []
    with open(file_path, 'r', encoding='iso-8859-1') as f:
        for line in f:
            if line.startswith('+++$+++'):
                conversation_parts = line.strip().split('+++$+++')
                if len(conversation_parts) == 4:
                    conversation_id = conversation_parts[0].strip()
                    conversation_lines = conversation_parts[3].strip()[1:-1].replace("'", "").replace(" ", "").split(",")
                    conversations.append(conversation_lines)

        conversations = self.extract_conversations(lines, conversations)

        return conversations

        # Separate questions and answers
        questions = []
        answers = []
        for conversation in conversations:
            for i in range(len(conversation) - 1):
                questions.append(conversation[i])
                answers.append(conversation[i + 1])

        # Generate encodings for the input and output sequences
        input_seqs, output_seqs = self.generate_encodings(questions, answers, self.max_length)

        # Save the tokenizer to disk
        self.save_tokenizer(tokenizer)

        return input_seqs, output_seqs, tokenizer.word_index

    def save_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Saves the tokenizer to disk.

        Args:
            tokenizer (Tokenizer): The tokenizer to save.
        """
        with open(self.tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_pairs(self, conversations: List[List[str]]) -> List[Tuple[str, str]]:
        """Generates question-answer pairs from the input conversations.

        Args:
            conversations (list): A list of conversations in the format [['line_1', 'line_2', ...], ...].

        Returns:
            list: A list of question-answer pairs in the format [(question_1, answer_1), ...].
        """
        pairs = []
        for conversation in conversations:
            for i in range(len(conversation) - 1):
                pairs.append((conversation[i], conversation[i + 1]))
        return pairs

    def split_dataset(self, pairs: List[Tuple[str, str]], validation_split: float) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Splits the input pairs into training and validation sets.

        Args:
            pairs (list): A list of question-answer pairs in the format [(question_1, answer_1), ...].
            validation_split (float): The percentage of pairs to use for validation.

        Returns:
            tuple: A tuple of lists of training and validation pairs.
        """
        # Shuffle the pairs
        np.random.shuffle(pairs)

        # Split the pairs into training and validation sets
        split_idx = int(len(pairs) * (1 - validation_split))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        return train_pairs, val_pairs

    def write_to_file(self, pairs: List[Tuple[str, str]], file_path: str) -> None:
        """Writes the input pairs to a file.

        Args:
            pairs (list): A list of question-answer pairs in the format [(question_1, answer_1), ...].
            file_path (str): The path to the file to write the pairs to.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(pair[0] + '\t' + pair[1] + '\n')

    def load_tokenizer(self) -> Tokenizer:
        """Loads the tokenizer from disk.

        Returns:
            Tokenizer: The loaded tokenizer.
        """
        with open(self.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def tokenize_text(self, text: str) -> List[int]:
        """Tokenizes the input text.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of token IDs.
        """
        tokenizer = self.load_tokenizer()
        tokens = tokenizer.texts_to_sequences([text])
        return tokens[0]

    def generate_response(self, input_text: str, model: Model, tokenizer: Tokenizer, max_length: int = 50) -> str:
        """
        Generates a response to the input text using the provided model.

        Args:
            input_text (str): The input text for which to generate a response.
            model (keras.Model): The model to use for generating the response.
            tokenizer (keras.preprocessing.text.Tokenizer): The tokenizer used to generate the input sequences.
            max_length (int, optional): The maximum sequence length to use for the input sequences. Defaults to 50.

        Returns:
            str: The generated response.
        """
        # Preprocess the input text
        input_text = self.preprocess_input_text(input_text)

        # Encode the input sequence
        input_sequence = tokenizer.texts_to_sequences([input_text])
        input_sequence = pad_sequences(input_sequence, maxlen=max_length, padding='post')

        # Generate the output sequence
        output_sequence = model.predict(input_sequence).argmax(axis=-1)

        # Decode the output sequence
        response = tokenizer.sequences_to_texts(output_sequence)[0]
        response = self.postprocess_response(response)

        return response
        # Decode the output sequence
    output_text = ''
    for token in output_sequence[1:]:
        if token == self.tokenizer.word_index['<end>']:
            break
        else:
            output_text += self.tokenizer.index_word[token] + ' '

    return output_text

    def train(self, epochs: int = 10, batch_size: int = 32, save_model: bool = True, save_path: str = 'model_weights.pt'):
        """
        Trains the model on the loaded dataset.

        Args:
            epochs (int, optional): The number of epochs to train for. Defaults to 10.
            batch_size (int, optional): The batch size to use during training. Defaults to 32.
            save_model (bool, optional): Whether or not to save the trained model. Defaults to True.
            save_path (str, optional): The file path to save the trained model weights to. Defaults to 'model_weights.pt'.

        Returns:
            None
        """
        # Compile the model
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

        # Fit the model on the data
        self.model.fit(
            self.encoder_input_data, self.decoder_output_data,
            batch_size=batch_size, epochs=epochs,
            validation_split=self.validation_split
        )

        # Save the trained model weights
        if save_model:
            self.model.save_weights(save_path)

    def load(self, weights_path: str):
        """
        Loads the trained model weights from file.

        Args:
            weights_path (str): The file path to load the model weights from.

        Returns:
            None
        """
        self.model.load_weights(weights_path)

    def generate_response(self, input_text: str, model: Optional[Model] = None) -> str:
        """
        Generates a response to the input text using the trained model.

        Args:
            input_text (str): The input text to generate a response for.
            model (Optional[Model], optional): The trained model to use for generating the response. If None, uses the
                                                model stored in the Seq2SeqModel object. Defaults to None.

        Returns:
            str: The generated response text.
        """
        # Use the stored model if one is not provided
        if not model:
            model = self.model

        # Encode the input text
        input_seq = self.tokenizer_inputs.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_input_len, padding='post')

        # Generate the decoder input sequence
        decoder_input_seq = np.zeros((1, self.max_output_len))
        decoder_input_seq[0, 0] = self.tokenizer_outputs.word_index['<start>']

        # Generate the output sequence
        output_seq = model.predict([input_seq, decoder_input_seq])

        # Decode the output sequence
        output_text = ''
        for i in range(len(output_seq[0])):
            predicted_index = np.argmax(output_seq[0][i])
            if predicted_index == 0:
                continue
            predicted_word = self.tokenizer_outputs.index_word[predicted_index]
            if predicted_word == '<end>':
                break
            output_text += ' ' + predicted_word

        return output_text.strip()



