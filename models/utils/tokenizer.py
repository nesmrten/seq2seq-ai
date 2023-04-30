import json
import os

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, sentence):
        return [self.word2idx[word] for word in sentence.split()]

    def decode(self, indices):
        return ' '.join([self.idx2word[idx] for idx in indices])

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f)
        print(f'Saved tokenizer to {file_path}')

    @staticmethod
    def load(file_path):
        tokenizer = Tokenizer()
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                tokenizer.word2idx = data['word2idx']
                tokenizer.idx2word = data['idx2word']
        else:
            print(f"Tokenizer file '{file_path}' does not exist. Creating new file...")
            tokenizer.save(file_path)
        return tokenizer
