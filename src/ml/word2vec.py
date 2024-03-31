import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SkipGramModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        nn.init.xavier_normal_(self.embeddings.weight)

        self.layer2 = nn.Linear(hidden_size, vocab_size)
        nn.init.xavier_normal_(self.layer2.weight)

    def forward(self, x) -> torch.Tensor:
        x = self.embeddings(x)
        x = self.layer2(F.relu(x))
        return x

    def predict(self, x) -> torch.Tensor:
        with torch.no_grad():
            x = self.forward(x)
        return x
    
    def get_embedding(self, idx: torch.Tensor) -> nn.Embedding:
        return self.embeddings(idx)

class SkipGramDataset(Dataset):
    def __init__(self, corpus, vocab_size, word_to_idx, idx_to_word, window_size=2, device = None):
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.window_size = window_size
        self.device = device
        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        for i, word in enumerate(self.corpus):
            lower_bound = max(0, i - self.window_size)
            upper_bound = min(i + self.window_size + 1, len(self.corpus))
            context_words = (
                self.corpus[lower_bound:i] + self.corpus[i + 1 : upper_bound]
            )
            for ctx_word in context_words:
                pairs.append((self.word_to_idx[word], self.word_to_idx[ctx_word]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]
