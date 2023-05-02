import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.utils.tokenizer import Tokenizer


def pad_collate_fn(batch):
    inputs = [torch.LongTensor(pair[0]) for pair in batch]
    targets = [torch.LongTensor(pair[1]) for pair in batch]

    inputs_lengths = torch.LongTensor([len(pair[0]) for pair in batch])
    targets_lengths = torch.LongTensor([len(pair[1]) for pair in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return inputs, inputs_lengths, targets, targets_lengths


class ChatbotDataset(Dataset):
    def __init__(self, data_dir, vocab_file, min_word_freq=2, min_length=1, max_length=50):
        self.data = []
        self.tokenizer = Tokenizer(vocab_size=10000)
        self.load_data(data_dir, vocab_file, min_word_freq, min_length, max_length)

    def load_data(self, data_dir, vocab_file, min_word_freq, min_length, max_length):
        with open(os.path.join(data_dir, "movie_lines.txt"), "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        id2line = {}
        for line in lines:
            parts = line.strip().split(" +++$+++ ")
            id2line[parts[0]] = parts[-1]

        with open(os.path.join(data_dir, "movie_conversations.txt"), "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(" +++$+++ ")
            conversation = [id2line[conv_id] for conv_id in parts[-1][1:-1].split(", ")]
            for i in range(len(conversation) - 1):
                input_line = conversation[i].strip()
                target_line = conversation[i + 1].strip()
                if min_length <= len(input_line.split()) <= max_length and \
                        min_length <= len(target_line.split()) <= max_length:
                    self.data.append((input_line, target_line))

        self.tokenizer.load_vocab_from_file(vocab_file)
        self.tokenizer.build_vocab_from_sentences([pair[0] for pair in self.data],
                                                   min_word_freq=min_word_freq)

    def __getitem__(self, index):
        input_sentence, target_sentence = self.data[index]
        input_tokens = self.tokenizer.tokenize_sentence(input_sentence)
        target_tokens = self.tokenizer.tokenize_sentence(target_sentence)
        input_ids = self.tokenizer.tokens2ids(input_tokens)
        target_ids = self.tokenizer.tokens2ids(target_tokens)
        return input_ids, target_ids

    def __len__(self):
        return len(self.data)

    def pad_collate_fn(batch):
        inputs = [torch.LongTensor(pair[0]) for pair in batch]
        targets = [torch.LongTensor(pair[1]) for pair in batch]

        inputs_lengths = torch.LongTensor([len(pair[0]) for pair in batch])
        targets_lengths = torch.LongTensor([len(pair[1]) for pair in batch])

        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

        return inputs, inputs_lengths, targets, targets_lengths
