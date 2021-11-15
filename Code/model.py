import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext


class FastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, text): 
        
        embedded = self.embedding(text)
        
        embedded = embedded.permute(1, 0, 2)
       
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        
        return self.fc(pooled)
    

