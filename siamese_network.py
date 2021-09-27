import torch
import torch.nn as nn
from dataset_loader import DatasetMaper, DatasetMaperList, DatasetMaperList2
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ray import tune

class LSTMEncoder(nn.ModuleList):
    def __init__(self, input_size, embedding_dim, lstm_layers, pretrained_model,
                 lstm_hidden_size=20, bidirectional=False,
                 lstm_dropout = 0.2, batch_size=32):
        super(LSTMEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.LSTM_layers = lstm_layers
        self.LSTM_hidden_size = lstm_hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.lstm_dropout = lstm_dropout
        self.batch_size = batch_size
        self.multiplication = 2 if bidirectional else 1
       
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Using GPU...")
            self.device = torch.device("cuda")
            self.to(self.device)
            
        print("Using pre-trained embedding")
        self.weights = torch.FloatTensor(pretrained_model.vectors)
        self.weights.requires_grad = False
        self.embedding = nn.Embedding.from_pretrained(self.weights)
        self.pretrained_embedding = True
            
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.LSTM_hidden_size,
                            num_layers=self.LSTM_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional,
                            dropout=self.lstm_dropout)
        
        def get_h_c(self):
            h = torch.zeros((self.LSTM_layers*self.multiplication, self.batch_size, self.LSTM_hidden_size)).to(self.device)
            c = torch.zeros((self.LSTM_layers*self.multiplication, x.size(0), self.LSTM_hidden_size)).to(self.device)
            return h, c
        
        def forward(self, x, hidden, cell):
            out = self.embedding(x)
            out, (h, c) = self.lstm(out, (hidden,cell))
            return out, h, c
        
class SiameseDetector(nn.ModuleList):
     def __init__(self, input_size, embedding_dim, batch_size, lstm_layers, pretrained_model, lstm_hidden_size, nn_size, fc_drouput):
         super(SiameseDetector, self).__init__()
         self.fc_drouput = fc_drouput
         self.nn_size = nn_size
         
         self.encoder = LSTMEncoder(input_size, embedding_dim, batch_size, lstm_layers, pretrained_model, lstm_hidden_size=lstm_hidden_size)
         
         self.fc1 = nn.Linear(in_features=lstm_hidden_size*2, out_features=self.nn_size)
         self.fc2 = nn.Linear(self.nn_size, 1)
        
    def forward(self, x, tweet1, tweet2): 
        
