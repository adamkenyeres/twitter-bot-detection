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
#https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python

class Model:
    @staticmethod
    def plot_accuracies(val_acc, train_acc, name):
        epochs = range(1, len(train_acc) + 1)
        plt.plot(epochs, train_acc, label='Training Accuracy', color='#377eb8')
        plt.plot(epochs, val_acc, label='Validation Accuracy', color='#a65628')
        plt.title(f'Validation and Training Accuracy for {name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.savefig(name + 'train_val_accuracy' + '.png')
        plt.show()

    def plot_losses(val_loss, train_loss, name):
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, val_loss, label='Validation Loss', color='#377eb8')
        plt.plot(epochs, train_loss, label='Training Loss', color='#a65628')
        plt.title(f'Validation and Training Loss for {name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.savefig(name + 'train_val_loss' + '.png')
        plt.show()
        
class TweetClassifier(nn.ModuleList):
    def __init__(self, input_size, embedding_dim, batch_size, lstm_layers,
                 preprocessor, lr, lstm_hidden_size=20, bidirectional=False,
                 pretrained_model=0, name='Tweet Classifier', weight_decay=1e-6, 
                 lstm_dropout = 0.2, fc_drouput=0.2, nn_size = 64, optimize=False, n_eopchs_stop = 14):
        super(TweetClassifier, self).__init__()
        print('Tweet classifier')
        self.name = name
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.LSTM_layers = lstm_layers
        self.LSTM_hidden_size = lstm_hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.preprocessor = preprocessor
        self.lr = lr
        self.weight_decay = weight_decay
        self.lstm_dropout = lstm_dropout
        self.fc_drouput = fc_drouput
        self.nn_size = nn_size
        self.optimize = optimize
        self.multiplication = 2 if bidirectional else 1
        self.n_eopchs_stop = n_eopchs_stop
       
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Using GPU...")
            self.device = torch.device("cuda")
            self.to(self.device)
            
        if pretrained_model == 0:
            self.embedding = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0)
            self.pretrained_embedding = False
        else:
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
        
        self.fc1 = nn.Linear(in_features=self.LSTM_hidden_size*self.multiplication, out_features=self.nn_size)
        self.fc2 = nn.Linear(self.nn_size, 1)
        self.dropout = nn.Dropout(self.fc_drouput)

    def forward(self, x):
        h = torch.zeros((self.LSTM_layers*self.multiplication, x.size(0), self.LSTM_hidden_size)).to(self.device)
        c = torch.zeros((self.LSTM_layers*self.multiplication, x.size(0), self.LSTM_hidden_size)).to(self.device)
        
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        
        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out, hidden.squeeze()
    # [1000, 100, 32]
    def get_prediction(self, x, y, loss_function = F.binary_cross_entropy):
        test_set = DatasetMaper(x, y)
        loader_test = DataLoader(test_set)
        predictions = []
        hidden_states = []
        probabilities = []
        total_loss = 0.0
        loss_steps = 0
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                x = x_batch.type(torch.LongTensor)
                x = x.to(self.device)
                y = y_batch.type(torch.FloatTensor)
                y = y.to(self.device)
                
                y_pred, h = self(x)
                h = h.cpu().detach().numpy()
                hidden_states.append(h[1]) # take hidden states from the last layer
                predictions.extend(np.round(y_pred.cpu().detach().numpy()))
                probabilities.extend(y_pred.cpu().detach().numpy())

                loss = loss_function(y_pred.reshape(-1), y)
                total_loss += loss.cpu().numpy()
                loss_steps +=1
                
                
        return predictions, hidden_states, probabilities, total_loss/loss_steps
    
    def send_optimization_result(self, epochs, val_loss, val_acc, train_accuracy, train_loss):
        if self.optimize:
            tune.report(iterations=epochs, mean_loss=val_loss, val_acc = val_acc, train_acc = train_accuracy, train_loss = train_loss)
            
    def train_model(self, x_train, y_train, x_val, y_val, epochs=10):
        print(self)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Using GPU...")
            device = torch.device("cuda")
            self.to(device)
        
        if self.pretrained_embedding:
            print('Using pretrained embedding')
        else:
            x_train = self.preprocessor.fit_transform(x_train)
            x_val = self.preprocessor.transform(x_val)
            
          
        training_set = DatasetMaper(x_train, y_train)
        loader_training = DataLoader(training_set, batch_size=self.batch_size)
        
        #optimizer =  optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer =  optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_function = nn.BCELoss()
        
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        num_epochs_not_improve = 0
        min_val_loss = np.Inf
        print('Starting to train the model')
        for epoch in range(epochs):
            print(f'Training epoch {epoch}')
            predictions = []
        
            self.train()
            train_loss, train_steps = 0, 0
            for x_batch, y_batch in loader_training:
                x = x_batch.type(torch.LongTensor)
                x = x.to(self.device)
                y = y_batch.type(torch.FloatTensor)
                y = y.to(self.device)
                y_pred,_ = self(x)
                y_pred = y_pred.reshape(-1)
                loss = loss_function(y_pred, y)
        
                optimizer.zero_grad()
        
                loss.backward()
        
                optimizer.step()
        
                predictions += list(np.round(y_pred.cpu().detach().numpy()))
                train_loss += loss.item()
                train_steps += 1
                
            train_loss = train_loss/train_steps
            train_accuracy = accuracy_score(y_train, predictions)
            
            val_pred, _, _, val_loss = self.get_prediction(x_val, y_val, loss_function)
            val_acc = accuracy_score(y_val, val_pred)
            print(f'Epoch: {epoch} Training loss: {train_loss} Training Accuracy {train_accuracy} Validation Accuracy: {val_acc}, Validation loss: {val_loss}')
            
            print('{{"metric": "Tweet Classifier Loss", "value": {}, "epoch": {}}}'.format(train_loss/train_steps, epoch))
            print('{{"metric": "Tweet Classifier accuracy", "value": {}, "epoch": {}}}'.format(val_acc, epoch))
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            self.send_optimization_result(epochs, val_loss, val_acc, train_accuracy, train_loss)
            
            if val_loss > min_val_loss:
                num_epochs_not_improve +=1
                print(f'Possibly early stopping will be neccesary {num_epochs_not_improve}')
            else:
                print('Re-setting early stopping')
                min_val_loss = val_loss
                num_epochs_not_improve = 0
                
            if epoch > 5 and num_epochs_not_improve == self.n_eopchs_stop:
                print('Early stopping')
                break;
            
        Model.plot_accuracies(val_accuracies, train_accuracies, self.name)
        Model.plot_losses(val_losses, train_losses, self.name)
            
        return self

class AccountClassifier(nn.ModuleList):
    def __init__(self, input_size, batch_size, lstm_layers, lr, weight_decay=0, 
                 lstm_dropout=0, fc_drouput=0, lstm_hidden_size=20, 
                 nn_size=128, bidirectional=False, optimize=False, n_eopchs_stop=20):
        super(AccountClassifier, self).__init__()
        
        self.name = 'Account Classifier'
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.lr = lr
        self.weight_decay = weight_decay
        self.lstm_dropout = lstm_dropout
        self.fc_drouput = fc_drouput
        self.nn_size = nn_size
        self.optimize = optimize
        self.n_eopchs_stop = n_eopchs_stop
        self.multiplication = 2 if bidirectional else 1
        
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Using GPU...")
            self.device = torch.device("cuda")
            self.to(self.device)
        
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            batch_first=False,
                            bidirectional=self.bidirectional, 
                            dropout=self.lstm_dropout)
        
        self.fc1 = nn.Linear(in_features=self.lstm_hidden_size*self.multiplication, out_features=self.nn_size)
        self.fc2 = nn.Linear(self.nn_size, 1)
        self.dropout = nn.Dropout(self.fc_drouput)

    def forward(self, x, length):
        h = torch.zeros((self.lstm_layers*self.multiplication, x.size(0), self.lstm_hidden_size)).to(self.device)
        c = torch.zeros((self.lstm_layers*self.multiplication, x.size(0), self.lstm_hidden_size)).to(self.device)
        
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        
        x = torch.nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        out, (hidden, cell) = self.lstm(x, (h,c))
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:,-1,:]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))

        return out, hidden.squeeze()
    
    def get_prediction(self, x, y, loss_function = F.binary_cross_entropy):
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        #Validation test sets will always have 100 tweets per account
        lengths = [100] * len(x)
        lengths = torch.Tensor(lengths)
        my_dataset = DatasetMaperList(x, y, lengths)
        loader_test = DataLoader(my_dataset)
        
        probabilities = []
        predictions = []
        total_loss = 0.0
        loss_steps = 0
        self.eval()
        with torch.no_grad():
            for x_batch, y_batch, lengths in loader_test:
                x = x_batch.type(torch.FloatTensor)
                x = x.to(self.device)
                y = y_batch.type(torch.FloatTensor)
                y = y.to(self.device)
    
                y_pred, _ = self(x, lengths)
                predictions.extend(np.round(y_pred.cpu().detach().numpy()))
                probabilities.extend(y_pred.cpu().detach().numpy())
                
                loss = loss_function(y_pred.reshape(-1), y)
                total_loss += loss.cpu().numpy()
                loss_steps +=1
                
        return predictions, total_loss/loss_steps, probabilities

    def send_optimization_result(self, epochs, val_loss, val_acc, train_accuracy, train_loss):
        if self.optimize:
            tune.report(iterations=epochs, mean_loss=val_loss, val_acc = val_acc, train_acc = train_accuracy, train_loss = train_loss)

    def train_model(self,  x_train, y_train, x_val, x_lengths, y_val, epochs=10):
        print(self)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Using GPU...")
            device = torch.device("cuda")
            self.to(device)
        
        x_train = torch.Tensor(x_train)
        x_lengths = torch.Tensor(x_lengths)
        y_train = torch.Tensor(y_train)
        my_dataset = DatasetMaperList(x_train, y_train, x_lengths)
        loader_training = DataLoader(my_dataset, batch_size=self.batch_size)
        
        optimizer =  optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay) #optim.Adam(classifier.parameters(), lr=LEARNING_RATE) #
        loss_function = F.binary_cross_entropy #nn.BCELoss()
        
        print('Starting to train the model')
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        num_epochs_not_improve = 0
        min_val_loss = np.Inf
        for epoch in range(epochs):
            print(f'Training epoch {epoch}')
            predictions = []
        
            self.train()
            train_loss, train_steps = 0, 0
            
            for x_batch, y_batch, length in loader_training:
                x = x_batch.type(torch.FloatTensor)
                x = x.to(self.device)
                y = y_batch.type(torch.FloatTensor)
                y = y.to(self.device)
                
                y_pred,_ = self(x, length)
                y_pred = y_pred.reshape(-1)
                loss = loss_function(y_pred, y)
        
                optimizer.zero_grad()
        
                loss.backward()
        
                optimizer.step()
        
                predictions += list(np.round(y_pred.cpu().detach().numpy()))
                
                train_loss += loss.item()
                train_steps += 1
                
            train_loss = train_loss/train_steps
            train_accuracy = accuracy_score(y_train, predictions)  
            
            val_pred, val_loss, _ = self.get_prediction(x_val, y_val, loss_function)
            val_acc = accuracy_score(y_val, val_pred)
            
            print(f'Epoch: {epoch} Training loss: {train_loss} Training Accuracy {train_accuracy} Validation Accuracy: {val_acc}, Validation loss: {val_loss}')

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            
            self.send_optimization_result(epochs, val_loss, val_acc, train_accuracy, train_loss)
            
            if val_loss > min_val_loss:
                num_epochs_not_improve +=1
                print(f'Possibly early stopping will be neccesary {num_epochs_not_improve}')
            else:
                print('Re-setting early stopping')
                min_val_loss = val_loss
                num_epochs_not_improve = 0
                
            if epoch > 5 and num_epochs_not_improve == self.n_eopchs_stop:
                print('Early stopping')
                break;
            
        Model.plot_accuracies(val_accuracies, train_accuracies, self.name)
        Model.plot_losses(val_losses, train_losses, self.name)
        
        return self