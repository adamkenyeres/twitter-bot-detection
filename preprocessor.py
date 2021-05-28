import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
'''nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')'''
import unicodedata
import re
import string
import contractions
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import truecase
from ekphrasis.classes.segmenter import Segmenter
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
import gc
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
#import spacy
#nlp = spacy.load("en_core_web_sm")
import en_core_web_sm

nlp = en_core_web_sm.load()
segmenter = Segmenter(corpus="twitter")

FLAGS = re.MULTILINE | re.DOTALL

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def remove_html(self, text):
        bs = BeautifulSoup(text, 'html.parser')
        return bs.get_text()

    def remove_accented_chars(self, text):
        new_text = unicodedata.normalize('NFKD', text).encode('ascii',
                                                              'ignore').decode(
                                                                  'utf-8',
                                                                  'ignore')
        return new_text

    def remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        #word_tokens = word_tokenize(text)
        word_tokens = text.split()
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    def remove_custom_stopwords(self, text):
        stop_words = []
        #xword_tokens = word_tokenize(text)
        word_tokens = text.split()
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return ' '.join(filtered_sentence)

    def replace_urls(self, text):
        pattern_url = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', re.I)
        text = pattern_url.sub('<url>', text)
        return text

    '''def remove_special_chars(self, text):
        pattern_special = re.compile(r'[^#,^a-zA-z0-9.,!?/:;\"\'\s]')
        text = pattern_special.sub('', text)
        return text'''

    def replace_special_chars(self, text):
        text = re.sub(r"/"," / ", text, flags=FLAGS)
        text = re.sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2", text, flags=FLAGS)
        text = re.sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )", text, flags=FLAGS)
        return text

    def replace_mentions(self, text):
        pattern_special = re.compile(r'@\w+')
        text = pattern_special.sub('<user>', text)
        return text

    def segment_text(self, text):
        return segmenter.segment(text)
    
    def replace_hashtags(self, text):
        tag = '<hashtag> '
        hashtags = re.findall("#\w+", text)
        for hashtag in hashtags:
            word = hashtag[1:]
            new_word = tag + self.segment_text(word)
            text = text.replace(hashtag, new_word)
        return text
    
    #https://www.kaggle.com/amackcrane/python-version-of-glove-twitter-preprocess-script
    def replace_emojis(self, text):
        eyes = r"[8:=;]"
        nose = r"['`\-]?"
        text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", text, flags=FLAGS)
        text = re.sub(r"{}{}p+".format(eyes, nose), "<lolface>", text, flags=FLAGS)
        text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", text, flags=FLAGS)
        text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", text, flags=FLAGS)
        text = re.sub(r"<3","<heart>", text, flags=FLAGS)
        return text
    
    def remove_repeats(self, text):
        text = re.sub(r"([!?.]){2,}", r"\1 <repeat>", text, flags=FLAGS)
        text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", text, flags=FLAGS)
        return text
    
    def replace_named_entities(self, text):
        doc = nlp(text)
        newString = text
        for e in reversed(doc.ents): 
            start = e.start_char
            end = start + len(e.text)
            newString = newString[:start] + e.label_ + newString[end:]
        return newString
    
    def remove_punctuation(self, text):
        new_text = ''
        for c in text:
            if c == '"':
                new_text = new_text +' ' + c +' '
            else:
                if c not in string.punctuation or c == '<' or c == '>':
                    new_text = new_text + c
        return new_text

    def replace_numbers(self, text):
        new_words = []
        for w in text.split():
            if w.isdigit():
                new_words.append('<number>')
            else:
                new_words.append(w)
        return ' '.join(new_words)

    def apply_stemming(self, text):
        lemmatizer = SnowballStemmer('english')
        new_words = []
        for w in word_tokenize(text):
            new_words.append(lemmatizer.stem(w))

        return ' '.join(new_words)

    def lemmatize(self, text):
        lemma = nltk.wordnet.WordNetLemmatizer()
        new_words = []
        #for w in word_tokenize(text):
        for w in text.split():
            new_words.append(lemma.lemmatize(w, pos='v'))
        return ' '.join(new_words)

    def remove_spaces(self, text):
        return ' '.join(text.split())

    def remove_space_before_special_char(self, text):
        return re.sub(r" >", r">", text, flags=FLAGS)
        
    def lower_acronyms(self, text):
        acronyms = ['rt', 'mt', 'dm', 'rofl', 'stfu', 'yolo', 'lmfao', 'nvm','ikr', 'ocf']
        new_words = []
        for w in text.split():
            if w.lower() in acronyms:
                new_words.append(w.lower())
            else:
                new_words.append(w)
        return ' '.join(new_words)
    
    def add_allcaps(self, text):
        tag = " <allcaps> "
        allcaps = re.findall(r" ([A-Z]{2,}) ", text)
        allcaps = sorted(allcaps, key=len, reverse=True)
        for allcap in allcaps:
            new_word = allcap.lower() + tag
            text = text.replace(allcap, new_word)
        return text
    
    def preprocess_text(self, text):
        text = self.remove_spaces(text)
        text = self.remove_html(text)
        text = self.replace_urls(text)
        text = self.replace_mentions(text)
        text = self.replace_hashtags(text)
        text = self.replace_numbers(text)
        text = self.replace_emojis(text)
        text = self.remove_accented_chars(text)
        text = self.remove_repeats(text)
        text = contractions.fix(text)
        text = self.add_allcaps(text)
        #text = truecase.get_true_case(text)
        #text = self.lower_acronyms(text)
        #text = self.replace_named_entities(text)
        text = self.replace_special_chars(text)
        text = text.lower()
        #text = self.remove_punctuation(text)
        text = self.remove_spaces(text)
        #text = self.lemmatize(text)
        return text.lower()

    def preprocess_target(self, df, target_label_name):
        df.loc[df[target_label_name] == 'human', target_label_name] = int(0)
        df.loc[df[target_label_name] == 'bot', target_label_name] = int(1)
        return df

    def preprocess_df(self, df, text_column_name, target_label_name):
        df = self.preprocess_text(df, text_column_name)
        print('Succesfully preprocessed the descriptive features')

        return self.preprocess_target(df, target_label_name)

    def split(self, df, text_column_name, target_label_name):
        train_x, test_x, train_y, test_y = train_test_split(
            df[text_column_name],
            df[target_label_name],
            test_size=0.2,
            random_state=42)

        train_x, val_x, train_y, val_y = train_test_split(train_x,
                                                          train_y,
                                                          test_size=0.2,
                                                          shuffle=True)
        train_y = np.array(train_y).astype('float32')
        test_y = np.array(test_y).astype('float32')
        val_y = np.array(val_y).astype('float32')

        return train_x, test_x, val_x, train_y, test_y, val_y
    
    def fit_transform(self, data, max_words = 1000, maxlen=20):
        self.tokenizer = Tokenizer(num_words=max_words)
        self.tokenizer.fit_on_texts(data)
        seqs = self.tokenizer.texts_to_sequences(data)
        return sequence.pad_sequences(seqs, maxlen=maxlen)

    def transform(self, data, maxlen=20):
        seqs = self.tokenizer.texts_to_sequences(data)
        return sequence.pad_sequences(seqs, maxlen=maxlen)
    
    def get_indexes(self, embedding, sentence):
        indexes = []
        for w in sentence.split():
            if embedding.has_index_for(w):
                indexes.append(embedding.get_index(w))
        return indexes
    
    def scale_hidden_states(self, accounts, should_fit=True):
        row_list = []
        names = []

        for i in range(0, len(accounts[0][0])):
            names.append(f'h_{i}')
            
        for account in accounts:
            for tweet in account:
              row_list.append(tweet)

        if should_fit:
            scalled_rows = self.scaler.fit_transform(row_list)
        else:
            scalled_rows = self.scaler.transform(row_list)
               
        curr_index = 0
        normalized_accounts = []
        for account in accounts:
            normalized_account = []
            for index in range(curr_index, curr_index + len(account)):
                normalized_account.append(scalled_rows[index])
            curr_index += len(account)
            normalized_accounts.append(normalized_account)
        
        return normalized_accounts
