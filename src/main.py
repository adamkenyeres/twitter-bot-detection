from preprocessor import Preprocessor
from data_understanding import DataUnderstanding
from lstm import TweetClassifier, AccountClassifier
from optimizer import Optimizer
from dataset_loader import DatasetMaper, DataSetLoader, DatasetMaperList, DatasetMaperList2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score, confusion_matrix
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
from keras.preprocessing import sequence
import gensim.models.fasttext as fasttext
import pickle
import pandas as pd
import numpy as np
import dataset_loader
from hyperopt import fmin, tpe, hp
from functools import partial
from pathlib import Path
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray import tune
from tweet_augmenter import TweetAugmenter

RUN_ON_SERVER = False
LOAD_PREPROCESSED_DATA = True
LOAD_TRAINED_CLASSIFIER = False
AUGMENT_TWEETS = True
if RUN_ON_SERVER:
    test_path = '/data/test/en/'
    train_path = '/data/training/en/'
    glove_path = "/data/glove.twitter.27B/glove.twitter.27B.50d.txt"
    #glove_out = '/glove/emb_word2vec_format.txt'
    glove_out = 'emb_word2vec_format.txt'
else:
    test_path = '../Data/test/en/'
    train_path = '../Data/training/en/'
    glove_path = '../Data/glove.twitter.27B/glove.twitter.27B.50d.txt'
    glove_out = '../Glove/emb_word2vec_format.txt'

earlybird_path = '../Data/earlybird/en/'

TWEET_LENGTH = 40
VOCABULARY_SIZE = 2000
LEARNING_RATE = 0.001
BATCH_SIZE = 64
ACCOUNT_TWEETS_NUMBER = 100
DATA_UNDERSTANDING_ENABLED = False
EMBEDDING_DIM = 50
tweet_classifier_path = 'tweet_classifier'
account_classifier_path = 'account_classifier'

def evaulate(y_pred, y):
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    print(f'Accuracy: {acc}')
    print(f'F1 Score: {f1}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    print("Classification Report:")
    print(classification_report(y, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    return f1

def get_tweet_based_account_prediction(x, y, classifier):
    accounts_pred = []
    accounts_probabilities = []
    accounts_hidden_states = []
    y_tweets = []
    y_tweets_probabilities = []
    i = 0
    for accounts in x:
        proccessed_percante = i/len(x)*100
        if i%100 == 0:
            print(f'Predicted {proccessed_percante}% of accounts')
        y_account = [y[i]]*len(accounts)
        #y_pred, h = classifier.get_prediction(classifier, accounts, y_account)
        y_pred, h, y_probabilites, _ = classifier.get_prediction(accounts, y_account)
        accounts_pred.append(y_pred)
        accounts_probabilities.append(y_probabilites)
        y_tweets.extend(y_pred)
        accounts_hidden_states.append(h)
        i += 1
    return accounts_pred, accounts_hidden_states, y_tweets #, accounts_probabilities

def get_majority_vote_for_account(account):
    number_of_bots = sum(account)
    return 1 if number_of_bots >= 50 else 0

def get_majority_vote(accounts):
    votes = []
    for account in accounts:
        votes.append(get_majority_vote_for_account(account))
    return votes

def write_to_file(data, name):
    output = open(name+'.pkl', 'wb')
    pickle.dump(data, output)

def get_from_file(name):
    pkl_file = open('serialization/' + name +'.pkl', 'rb')
    return pickle.load(pkl_file)

def get_x_y_tweets(x, y):
    x_tweets = []
    y_tweets = []
    for i in range(0, len(x)):
        x_tweets.extend([tweet for tweet in x[i]])
        y_tweets.extend([y[i]]*len(x[i]))
    return x_tweets, y_tweets

def save_preprocessed_data(x_test, y_test, x, y, x_val, y_val, hashes_val, hashes_test, hashes_train):
    print('Saving preprocessed data to Files')
    Path('serialization').mkdir(parents=True, exist_ok=True)
    write_to_file(x_test, 'serialization/x_test')
    write_to_file(y_test, 'serialization/y_test')
    write_to_file(x, 'serialization/x')
    write_to_file(y, 'serialization/y')
    write_to_file(x_val, 'serialization/x_val')
    write_to_file(y_val, 'serialization/y_val')

    write_to_file(hashes_test, 'serialization/hashes_test')
    write_to_file(hashes_train, 'serialization/hashes_train')
    write_to_file(hashes_val, 'serialization/hashes_val')

data_understander = DataUnderstanding()
preprocessor = Preprocessor()

glove2word2vec(glove_input_file=glove_path, word2vec_output_file=glove_out)
embedding = gensim.models.KeyedVectors.load_word2vec_format(glove_out)
dataset_loader = DataSetLoader(train_path)

if LOAD_PREPROCESSED_DATA:
    print('Getting serailized data\n')
    x_test = get_from_file('x_test')
    y_test = get_from_file('y_test')
    x = get_from_file('x')
    y = get_from_file('y')
    x_val = get_from_file('x_val')
    y_val = get_from_file('y_val')
    x_train_aug = get_from_file('x_train_aug')
    y_train_aug = get_from_file('y_train_aug')
    hashes_test = get_from_file('hashes_test')
    hashes_train = get_from_file('hashes_train')
    hashes_val = get_from_file('hashes_val')
    hashes_train_aug = get_from_file('hashes_train_aug')

    #x, y, x_train, y_train, x_val, y_val, x_padded_train, x_padded_val, x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val = dataset_loader.get_all_data_from_x_y(x ,y, embedding)
    x_test, y_test, x_padded_test, x_tweets_test, y_tweets_test = dataset_loader.get_all_data_from_x_y(x_test, y_test, embedding, shouldSplit=False)
    x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train = dataset_loader.get_all_data_from_x_y(x, y, embedding, shouldSplit=False)
    x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train = dataset_loader.get_all_data_from_x_y(x, y, embedding, shouldSplit=False)
    x_val, y_val, x_padded_val, x_tweets_val, y_tweets_val = dataset_loader.get_all_data_from_x_y(x_val, y_val, embedding, shouldSplit=False)
    x_train_aug, y_train_aug, x_padded_train_aug, x_tweets_train_aug, y_tweets_train_aug = dataset_loader.get_all_data_from_x_y(x_train_aug, y_train_aug, embedding, shouldSplit=False)
else:
    print('Getting training and test set')
    #x, y, x_train, y_train, x_val, y_val, x_padded_train, x_padded_val, x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val, hashes_train = dataset_loader.get_all_x_y(earlybird_path, embedding)
    x_test, y_test, x_padded_test, x_tweets_test, y_tweets_test, hashes_test = dataset_loader.get_all_x_y(test_path, embedding, shouldSplit=False)
    x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train, hashes_train = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-train.txt', shouldSplit=False)
    x_val, y_val, x_padded_val, x_tweets_val, y_tweets_val, hashes_val = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-dev.txt', shouldSplit=False)

    save_preprocessed_data(x_test, y_test, x_train, y_train, x_val, y_val, hashes_val, hashes_test, hashes_train)

#x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train, hashes_train = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-train.txt', shouldSplit=False, should_preprocess=False)
x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train, hashes_train = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-train.txt', shouldSplit=False)
print('Getting training and test set')
x_train, y_train, x_padded_train, x_tweets_train, y_tweets_train, hashes_train = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-train.txt', shouldSplit=False, should_preprocess=True)
x_test, y_test, x_padded_test, x_tweets_test, y_tweets_test, hashes_test = dataset_loader.get_all_x_y(test_path, embedding, shouldSplit=False, should_preprocess=False)
x_val, y_val, x_padded_val, x_tweets_val, y_tweets_val, hashes_val = dataset_loader.get_all_x_y(train_path, embedding, truth='truth-dev.txt', shouldSplit=False, should_preprocess=False)

x_train[0][0]
x_test[0][0]
write_to_file(x_test, 'x_test_without_prep')
write_to_file(y_test, 'y_test_without_prep')
write_to_file(x_val, 'x_val_without_prep')
write_to_file(y_val, 'y_val_without_prep')
x_tweets_train, y_tweets_train = get_x_y_tweets(x_train, y_train)
x_tweets_val, y_tweets_val = get_x_y_tweets(x_val, y_val)
x_tweets_test, y_tweets_test = get_x_y_tweets(x_test, y_test)

write_to_file(x_tweets_train, 'x_tweet_train_without_prep')
write_to_file(x_tweets_val, 'x_tweet_val_without_prep')
write_to_file(x_tweets_test, 'x_tweet_test_without_prep')

write_to_file(y_tweets_train, 'y_tweet_train_without_prep')
write_to_file(y_tweets_val, 'y_tweet_val_without_prep')
write_to_file(y_tweets_test, 'y_tweet_test_without_prep')

y_train[0]
bots_train = []
human_train = []
for i in range(0, len(x_train)):
    if i %2 == 0:
        human_train.extend(x_train[i])
    else:
        bots_train.extend(x_train[i])

bots_train[100]
human_train[100]

bots_y = len(bots_train)*[1]
human_y = len(human_train)*[0]

bots_y
print(x_val[100][0])
x_tweets, y_tweets = dataset_loader.get_x_y_tweets(x_train, y_train)
x_tweets_train[0]
y_tweets[0]
data_understander = DataUnderstanding()
data_understander.plot_tweet_token_distribution(x_train)
data_understander.plot_target_distribution(x_tweets_train, y_tweets_train)
data_understander.plot_most_frequent_tokens(bots_train, bots_y, 'Most Frequently used tokens by Bots')
data_understander.plot_most_frequent_tokens(human_train, human_y, 'Most Frequently used tokens by Humans')
data_understander.plot_special_char_distribution(x_tweets, y_tweets)
data_understander.tweet_contains_emoji(':)')

print(x_tweets_train[1000])
len(x_test)
if AUGMENT_TWEETS:
    tweet_augmenter = TweetAugmenter(embedding, glove_path, x = x_train, y=y_train, hash=hashes_train)
    augmented_x, augmented_y, augmented_hashes,  account_orig = tweet_augmenter.augment()
    write_to_file(augmented_x, 'serialization/x_train_aug')
    write_to_file(augmented_y, 'serialization/y_train_aug')
    write_to_file(augmented_hashes, 'serialization/hash_train_aug')

len(x_train)
x_train_aug = get_from_file('x_train_aug')
y_train_aug = get_from_file('y_train_aug')
hashes_train_aug = get_from_file('hash_train_aug')
y_train_aug[0]
x_train[1][0]
x_train_aug[1][0]

x_train_aug_preprocessed = []
for account in x_train_aug:
    prep_account = []
    for tweet in account:
        prep_account.append(preprocessor.preprocess_text(tweet))
    x_train_aug_preprocessed.append(prep_account)

print(len(x_train_aug_preprocessed))
x_train_augmented = x_train.copy()
x_train_augmented.extend(x_train_aug)

y_train_augmented = y_train.copy()
y_train_augmented.extend(y_train_aug)

hashes_train_augmented = hashes_train.copy()
hashes_train_augmented.extend(hashes_train_aug)
x_train_augmented[0][0]
x_train_augmented[2880][0]

len(y_train_augmented)


hashes_train_augmented[3880]

len(hashes_train_augmented)
hashes_train_augmented[2880]
y_train_aug[0]


x_train_augmented[2880][0]
y_train_augmented[2880]


len(y_train_augmented)
len(x_train_augmented)
len(hashes_train_augmented)
write_to_file(x_train_augmented, 'serialization/x_train_augmented')
write_to_file(y_train_augmented, 'serialization/y_train_augmented')
write_to_file(hashes_train_augmented, 'serialization/hashes_train_augmented')

print(len(x_train))
print(x_train[100][1])
print(y_train_aug[100])
print(x_train_aug[100][1])
x_train.extend(x_train_aug)
len(x_train)
preprocessor.preprocess_text(x_train_aug[100][1])
print('\n-------------Tweet Classification----------')
if LOAD_TRAINED_CLASSIFIER:
    print('Loading pre-trained model')
    tweet_classifier = torch.load('tweet_classifier',  map_location=torch.device('cpu'))
else:
    tweet_classifier = TweetClassifier(input_size=VOCABULARY_SIZE,
                                     embedding_dim = EMBEDDING_DIM,
                                     batch_size = BATCH_SIZE,
                                     lstm_layers = 2,
                                     preprocessor=preprocessor,
                                     lr=LEARNING_RATE,
                                     lstm_hidden_size=32,
                                     pretrained_model = embedding,
                                     bidirectional = True,
                                     weight_decay = 1e-6,
                                     dropout = 0.2,
                                     nn_size = 64)
    print('\nTraining Tweet classifier')
    tweet_classifier.train_model(x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val, epochs=10)

    print('Saving trained models')
    torch.save(tweet_classifier, 'tweet_classifier')

tweet_based_accounts_pred, accounts_h_train, y_tweets_pred, tweet_based_account_probabilites = get_tweet_based_account_prediction(x_padded_test, y_test, tweet_classifier)

accounts_h_train[100][0]

df, acc = preprocessor.scale_hidden_states(accounts_h_train)

print(acc[100][0])
print(df.loc[10000])
account_majority_vote = get_majority_vote(tweet_based_accounts_pred)
account_probability_vote = get_majority_vote(tweet_based_account_probabilites)

print('\nEvaulation of Tweet Prediction')
evaulate(y_tweets_pred, y_tweets_test)
evaulate(account_majority_vote, y_test)
evaulate(account_probability_vote, y_test)'''

'''
print('\n-------------Account Classification----------')

account_classifier = AccountClassifier(input_size=32,
                                     batch_size = BATCH_SIZE,
                                     lstm_layers = 2,
                                     lr = LEARNING_RATE,
                                     lstm_hidden_size=32,
                                     bidirectional = True)

account_classifier.train_model(accounts_h_train, y_test, epochs=100)
print('Saving account classifier')
torch.save(account_classifier, account_classifier_path)

print('\n-------------Testing----------')

y_tweet_test_pred, y_tweet_h, y_tweets, tweet_based_account_probabilites = get_tweet_based_account_prediction(x_padded_train, y_train, tweet_classifier)
y_tweet_test_pred, y_tweet_h, y_tweets, tweet_based_account_probabilites = get_tweet_based_account_prediction(x_padded_test, y_test, tweet_classifier)
y_account_test_pred = account_classifier.get_prediction(y_tweet_h, y_test)

account_majority_vote = get_majority_vote(y_tweet_test_pred)
account_probability_vote = get_majority_vote(tweet_based_account_probabilites)

print('\n-------Evaulation-------')

print('Majority Voting:')
evaulate(account_majority_vote, y_train)
evaulate(account_majority_vote, y_test)

print('\nAccount Classifier:')
evaulate(y_account_test_pred, y_test)

print('------Done with Training and Testing----------')
tweet_classifier = TweetClassifier(input_size=VOCABULARY_SIZE,
                                 embedding_dim = EMBEDDING_DIM,
                                 batch_size = BATCH_SIZE,
                                 lstm_layers = 2,
                                 preprocessor=preprocessor,
                                 lr=LEARNING_RATE,
                                 lstm_hidden_size=32,
                                 pretrained_model = embedding,
                                 bidirectional = True,
                                 weight_decay = 1e-6,
                                 lstm_dropout = 0.2,
                                 nn_size = 64)

account_classifier = AccountClassifier(input_size=32,
                                     batch_size = BATCH_SIZE,
                                     lstm_layers = 2,
                                     lr = LEARNING_RATE,
                                     lstm_hidden_size=32,
                                     bidirectional = True)

combined = CombinedClassifier(tweet_classifier, account_classifier)
x_train[0][0]
x_val[1][0]
y_train
x_padded_train[0][0]
combined.train_model(x_train=x_padded_train, y_train=y_train, x_val=x_val, y_val=y_val, epochs=10)

x_train = torch.Tensor(x_padded_train)
x_lengths = 0
y_train = torch.Tensor(y_train)
my_dataset = DatasetMaperList2(x_train, y_train)
loader_training = DataLoader(my_dataset, batch_size=32)
for x_batch, y_batch in loader_training:
    print(len(x_batch))
