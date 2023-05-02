import json
import os
from collections import Counter


class Vocabulary:
    def __init__(self, data_dir, vocab_file, min_word_frequency=2, special_tokens=None):
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.min_word_frequency = min_word_frequency
        self.special_tokens = special_tokens or ['<pad>', '<sos>', '<eos>', '<unk>']

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

        with open(os.path.join(self.data_dir, self.vocab_file), 'w', encoding='utf-8') as f:
            vocab_dict = {word: idx for idx, word in enumerate(vocab_list)}
            json.dump(vocab_dict, f)

        print('Vocabulary size:', len(vocab_list))
        print('Saved vocabulary to:', os.path.join(self.data_dir, self.vocab_file))


if __name__ == '__main__':
    import argparse

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Build a vocabulary from the Cornell Movie Dialogs Corpus.")
    parser.add_argument("data_dir", type=str, help="the directory containing the data")
    parser.add_argument("vocab_file", type=str, help="the name of the vocabulary file")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define the file paths
    movie_lines_path = os.path.join(args.data_dir, "movie_lines.txt")
    movie_conversations_path = os.path.join(args.data_dir, "movie_conversations.txt")

    # Load the lines and conversations
    with open(movie_lines_path, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
    with open(movie_conversations_path, 'r', encoding='iso-8859-1') as f:
        conversations = f.readlines()

    # Build the pairs of questions and answers
    pairs = []
    for conversation in conversations:
        line_ids = conversation.split(" +++$+++ ")[-1][1:-2].replace("'", "").split(", ")
        for i in range(len(line_ids) - 1):
            pairs.append((lines[int(line_ids[i])].strip(), lines[int(line_ids[i + 1])].strip()))

    # Build the vocabulary
    vocabulary = Vocabulary(args.data_dir, args.vocab_file)
    vocabulary.build_vocabulary(pairs)
