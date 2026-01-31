# DANmodels.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
from utils import Indexer


class SentimentDatasetDAN(Dataset):
    def __init__(self, sentences, labels, indexer):
        self.sentences = sentences
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.indexer = indexer

    def __getitem__(self, idx):
        words = self.sentences[idx]
        indices = [self.indexer.index_of(w) if self.indexer.contains(w) else 1 for w in words]  # 1 = UNK
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

    def __len__(self):
        return len(self.sentences)
    
class WordEmbeddings:
    def __init__(self, indexer, vectors):
        self.indexer = indexer            # your Indexer
        self.vectors = vectors            # numpy array (vocab_size, dim)

    def get_embedding(self, word):
        idx = self.indexer.index_of(word)
        if idx == -1:
            idx = self.indexer.index_of("UNK")
        return self.vectors[idx]

    def get_embedding_by_index(self, idx):
        return self.vectors[idx]

    def get_vocab_size(self):
        return len(self.indexer)

    def get_embedding_dim(self):
        return self.vectors.shape[1]
    
class DAN(nn.Module):
    def __init__(self, word_embeddings, embedding_dim, hidden_dim, num_classes, dropout=0.50):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding = self.get_initialized_embedding_layer(word_embeddings, frozen=False)

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        #x_embed = self.embedding(x)
        #mask = (x != 0).unsqueeze(-1)
        #x_embed = x_embed * mask
        #lengths = mask.sum(dim=1)  # (batch, 1)       # Compute the mean, divide by number of real words per sentence
        #x = x_embed.sum(dim=1) / lengths  # (batch, embedding_dim)
        x = torch.mean(x, dim=1)  # Average over the sequence length, back to (batch, seq_len)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)
    
    def get_initialized_embedding_layer(self, word_embeddings, frozen=False):
        embedding_layer = nn.Embedding(
            num_embeddings=word_embeddings.get_vocab_size(),
            embedding_dim=word_embeddings.get_embedding_dim(),
            padding_idx=0
        )

        embedding_layer.weight.data.copy_(torch.from_numpy(word_embeddings.vectors))

        embedding_layer.weight.requires_grad = not frozen
        return embedding_layer