import os
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config


class ChatbotDataset(Dataset):
    def __init__(self, data_dir, vocab_file, min_word_freq=Config().MIN_WORD_FREQ, min_length=Config().MIN_LENGTH, max_length=Config().MAX_LENGTH):
        self.data_dir = data_dir
        self.vocab_file = vocab_file
        self.min_word_freq = min_word_freq
        self.min_length = min_length
        self.max_length = max_length
        self.word2id, self.id2word = self.load_vocab()

        self.conversations = self.load_conversations()

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]

    def load_vocab(self):
        with open(self.vocab_file, "r") as f:
            vocab = [line.strip() for line in f.readlines()]
        word2id = {w: i for i, w in enumerate(vocab)}
        id2word = {i: w for i, w in enumerate(vocab)}
        return word2id, id2word

    def load_conversations(self):
        conversations = []
        with open(os.path.join(self.data_dir, "movie_lines.txt"), "r", encoding="iso-8859-1") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" +++$+++ ")
                if len(parts) == 5:
                    line_text = parts[4]
                    if len(line_text.split(" ")) >= self.min_length and len(line_text.split(" ")) <= self.max_length:
                        conversations.append(line_text)
        return conversations

    def preprocess(self, sentence):
        tokens = sentence.lower().split(" ")
        result = []
        for token in tokens:
            if token in self.word2id:
                result.append(self.word2id[token])
            else:
                result.append(self.word2id[Config.UNK_TOKEN])
        return result


def pad_collate_fn(batch, word2id):
    input_batch = []
    target_batch = []
    for conversation in batch:
        input_sentence = conversation[:-1]
        target_sentence = conversation[1:]
        input_tokens = preprocess(input_sentence, word2id)
        target_tokens = preprocess(target_sentence, word2id)
        input_batch.append(torch.LongTensor(input_tokens))
        target_batch.append(torch.LongTensor(target_tokens))
    input_batch = torch.nn.utils.rnn.pad_sequence(input_batch, padding_value=word2id[Config.PAD_TOKEN])
    target_batch = torch.nn.utils.rnn.pad_sequence(target_batch, padding_value=word2id[Config.PAD_TOKEN])
    return input_batch, target_batch


if __name__ == "__main__":
    dataset = ChatbotDataset("data/cornell movie-dialogs corpus", "data/cornell movie-dialogs corpus/vocab.txt")
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda batch: pad_collate_fn(batch, dataset.word2id))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        print(input_batch)
        print(target_batch)
        if i > 2:
            break
