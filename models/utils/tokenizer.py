import json
from collections import Counter
from itertools import chain

class Tokenizer:
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.stoi = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
        self.itos = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        self.counter = Counter()

    def fit(self, texts):
        self.counter = Counter(chain.from_iterable(texts))
        self.counter = Counter({k: c for k, c in self.counter.items() if c >= 5})
        vocab = [token for token in self.counter.keys()]
        for i, token in enumerate(vocab, 4):
            self.stoi[token] = i
            self.itos[i] = token

    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        return [self.stoi.get(token, self.stoi['<unk>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.itos.get(i, '<unk>') for i in ids]

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump({
                'max_len': self.max_len,
                'stoi': self.stoi,
                'itos': self.itos,
                'counter': self.counter,
            }, f)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        tokenizer = cls(max_len=data['max_len'])
        tokenizer.stoi = data['stoi']
        tokenizer.itos = data['itos']
        tokenizer.counter = Counter(data['counter'])
        return tokenizer
