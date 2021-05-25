from torch.utils.data import Dataset
import pandas as pd
import xml.etree.ElementTree as ET
from preprocessor import Preprocessor
import os
import numpy as np
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

class DatasetMaper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DatasetMaperList(Dataset):

    def __init__(self, x, y, length):
        self.x = x
        self.y = y
        self.length = length
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx], self.length[idx]

class DatasetMaperList2(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return np.array(self.x[idx]), self.y[idx]

class DataSetLoader():
    def __init__(self, path=''):
        self.path = path
        self.preprocessor = Preprocessor()
        
    def get_tweets(self, hash, should_preprocess=True):
        tweets = []
        tree = ET.parse(self.path + hash + '.xml')
        root = tree.getroot()
        for tweet in root.iter('document'):
            if should_preprocess:
                tweets.append(self.preprocessor.preprocess_text(tweet.text))
            else:
                tweets.append(tweet.text)
                
        return tweets

    def get_dict(self, truth='truth.txt', should_preprocess=True):
        print(f'Reading data from {self.path} using {truth}')
        accounts = {}
        bots = {}
        humans = {}
        truth = pd.read_csv(self.path + truth, sep=":::", header=None, engine='python')
        size = len(truth)
        for index, row in truth.iterrows():
            proccessed_percante = index/size*100
            if index%100 == 0:
                print(f'Proccesed {proccessed_percante}% of accounts')
            hash = row[0]
            tweets = self.get_tweets(hash, should_preprocess)
            accounts[hash] = tweets
    
            if row[1] == 'bot':
                bots[hash] = tweets
            else:
                humans[hash] = tweets
        return accounts, humans, bots
    
    
    def flatten_dict(self, dictionary):
        l = []
        for key, value in dictionary.items():
            l.extend(value)
        return l
    
    def write_dict_to_file(self, dic, name):
        os.mkdir(name)
        for key, value in dic.items():
            os.mkdir(name + '/' + key)
            print(len(value))
            for i in range(0, len(value)):
                f = open(name + '/' + key + '/' + str(i)+".txt", "w", encoding="utf-8")
                f.write(value[i])
                f.close()
    
    def get_df_humans_bots(self, truth='truth.txt', should_preprocess=True):
            
        accounts, humans, bots = self.get_dict(truth, should_preprocess)
        bot_list = self.flatten_dict(bots)
        human_list = self.flatten_dict(humans)
        
        identity = [1] * len(bot_list)
        identity.extend([0] * len(human_list))
        
        all_accounts = bot_list
        all_accounts.extend(human_list)
        
        df = pd.DataFrame()
        df['Tweet'] = all_accounts
        df['isBot'] = identity
        return df, humans, bots
    
    def get_x_y(self, humans, bots):
        x = []
        y = []
        hashes = []
        for (kh,vh), (kb, vb) in zip(humans.items(), bots.items()):
            x.append(humans[kh])
            x.append(bots[kb])
            y.append(0)
            y.append(1)
            hashes.append(kh)
            hashes.append(kb)
        
        return x, y, hashes

    def get_padded_x(self, x, embedding):
        x_padded = []
        for account in x:
            padded_tweets = []
            for tweet in account:
                padded_tweets.extend(sequence.pad_sequences([self.preprocessor.get_indexes(embedding, str(tweet))], maxlen=30))
            x_padded.append(padded_tweets)
        return x_padded
    
    def get_x_y_tweets(self, x_padded, y):
        x_tweets = []
        y_tweets = []
        for i in range(0, len(x_padded)):
            x_tweets.extend([tweet for tweet in x_padded[i]])
            y_tweets.extend([y[i]]*len(x_padded[i]))
        return x_tweets, y_tweets
    
    def get_all_data_from_x_y(self, x,y, embedding, shouldSplit=True):
        if shouldSplit:
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, shuffle=True)
            
            x_padded_train = self.get_padded_x(x_train, embedding)
            x_padded_val = self.get_padded_x(x_val, embedding)
            x_tweets_train, y_tweets_train = self.get_x_y_tweets(x_padded_train, y_train)
            x_tweets_val, y_tweets_val = self.get_x_y_tweets(x_padded_val, y_val)
            
            return x, y, x_train, y_train, x_val, y_val, x_padded_train, x_padded_val, x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val
        else:
            x_padded = self.get_padded_x(x, embedding)
            x_tweets, y_tweets = self.get_x_y_tweets(x_padded, y)
            
            return x, y, x_padded, x_tweets, y_tweets
        
    def get_all_x_y(self, path, embedding, shouldSplit=True, truth='truth.txt', should_preprocess=True):
        self.path = path
        data_frame, humans, bots = self.get_df_humans_bots(truth, should_preprocess=False)
        
        x, y, hashes = self.get_x_y(humans, bots)
        
        if shouldSplit:
            x, y, x_train, y_train, x_val, y_val, x_padded_train, x_padded_val, x_tweets_train, \
            y_tweets_train, x_tweets_val, y_tweets_val = self.get_all_data_from_x_y(x, y, embedding, shouldSplit)
            return x, y, x_train, y_train, x_val, y_val, x_padded_train, x_padded_val, x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val, hashes
        else:
            x, y, x_padded, x_tweets, y_tweets = self.get_all_data_from_x_y(x, y, embedding, shouldSplit)
            return x, y, x_padded, x_tweets, y_tweets, hashes