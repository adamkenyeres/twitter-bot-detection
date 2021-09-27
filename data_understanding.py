"""Module containing methods for data understanding."""

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import  CountVectorizer
from yellowbrick.text import FreqDistVisualizer
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
import numpy as np
import re

class DataUnderstanding:
    '''
    Resplonsible for plotting graphs of the data.'''
    def plot_target_distribution(self, x, y):
        d = {'Tweets': x, 'Label':y}
        df = pd.DataFrame(data=d)
        names = ['Human', 'Bot']
        sizes = [len(df[df['Label'] == 1]),
                 len(df[df['Label'] == 0])]
        #plt.bar(names, sizes, color ='maroon', width = 0.4)
        sns.barplot(x=names,y=sizes)
        plt.ylabel('Count')
        plt.title('Distribution of the Accounts')
        plt.savefig('target_distribution.png')
        plt.show()

    def plot_most_frequent_tokens(self, x, y, label):
        d = {'Tweets': x, 'Label':y}
        df = pd.DataFrame(data=d)
        count_vectorizer = CountVectorizer()
        tf_original = count_vectorizer.fit_transform(df['Tweets'])
        tf_feature_names = count_vectorizer.get_feature_names()
        visualizer = FreqDistVisualizer(features=tf_feature_names, orient='v', title=label)
        visualizer.fit(tf_original)
        print(tf_feature_names[:10])
        visualizer.show(outpath=label+'.svg')

    def plot_distribution(self, d, x_label='Number', y_label='Size', title='Distribution'):
        sns.displot(data=d, kde=True)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(title+'.png')
        plt.savefig(title + '.svg', bbox_inches='tight')
        plt.show()
        
    def plot_box_plot(self, d, title='Distribution'):
        sns.set_theme(style="whitegrid")
        sns.boxplot(y=d)
        plt.title(title)
        plt.savefig(title+'.png')
        plt.show()
        
    def plot_tweet_token_distribution(self, x):
        tweet_lengths = []
        for account in x:
            for tweet in account:
                tweet_lengths.append(len(tweet.split()))
        
        print(min(tweet_lengths))
        self.plot_distribution(tweet_lengths, 'Words', 'Size', 'Tweet Word Distribution')
        self.plot_box_plot(tweet_lengths, 'Box Plot for Tweet Words')
        
    def tweet_contains_emoji(self, tweet):
        FLAGS = re.MULTILINE | re.DOTALL
        eyes = r"[8:=;]"
        nose = r"['`\-]?"
        smile = re.compile(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), flags=FLAGS)
        lolface = re.compile(r"{}{}p+".format(eyes, nose), flags=FLAGS)
        sadface = re.compile(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), flags=FLAGS)
        neutralface = re.compile(r"{}{}[\/|l*]".format(eyes, nose), flags=FLAGS)
        heart = re.compile(r"<3", flags=FLAGS)
        
        if re.match(smile, tweet) or re.match(lolface, tweet) or re.match(sadface, tweet) or re.match(neutralface, tweet) or re.match(heart, tweet): return True 
        else: return False 
        
    def check_for_special_chars(self, tweet, mentions, hashstags, rts, emojis, url):
        pattern_url = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', re.I)
        if '@' in tweet: mentions +=1
        if '#' in tweet: hashstags +=1
        if 'RT' in tweet: rts +=1
        if re.match(pattern_url, tweet): url+=1
        if self.tweet_contains_emoji(tweet): emojis +=1
        return mentions, hashstags, rts, emojis, url
        
    def plot_special_char_distribution(self, x_tweets, y_tweets):
        bots_mentions = 0
        bots_hashtags = 0
        bots_rt = 0
        bots_url = 0
        bots_emoji = 0
      
        humans_mentions = 0
        humans_hashtags = 0
        humans_urls= 0
        humans_rt = 0
        humans_emoji = 0
        
        for i in range(0, len(x_tweets)):
            if y_tweets[i] == 0:
                humans_mentions, humans_hashtags, humans_rt, humans_emoji, humans_urls = self.check_for_special_chars(x_tweets[i], humans_mentions, humans_hashtags, humans_rt, humans_emoji, humans_urls)
            else:
                bots_mentions, bots_hashtags, bots_rt, bots_emoji, bots_url = self.check_for_special_chars(x_tweets[i], bots_mentions, bots_hashtags, bots_rt, bots_emoji, bots_url)
        
        human_values = []
        bot_values = []
        bar_width = 0.35
        
        human_values.append(100*humans_mentions/(humans_mentions+bots_mentions))
        human_values.append(100*humans_rt/(humans_rt+bots_rt))
        human_values.append(100*humans_hashtags/(humans_hashtags+bots_hashtags))
        human_values.append(100*humans_urls/(humans_urls + bots_url))
        human_values.append(100*humans_emoji/(humans_emoji + bots_emoji))
        index = np.arange(len(human_values))

        bot_values.append(100*bots_mentions//(humans_mentions+bots_mentions))
        bot_values.append(100*bots_rt/(humans_rt+bots_rt))
        bot_values.append(100*bots_hashtags/(humans_hashtags+bots_hashtags))
        bot_values.append(100*bots_url/(humans_urls + bots_url))
        bot_values.append(100*bots_emoji/(humans_emoji + bots_emoji))
        
        plt.bar(index, human_values, bar_width, label='Humans')
        plt.bar(index + bar_width, bot_values, bar_width, label='Bots')
        plt.xticks(index + bar_width/2, ('Mentions', 'RT', 'Hashtags', 'URls', 'Emojis'))
        plt.xlabel('Symbols')
        plt.ylabel('%')
        plt.title('Types of Tweets')
        plt.legend()
        plt.savefig('special_char_dist.png')
        plt.show()
        
        print(f'Humans with URL: {humans_urls}, Bots with URLs: {bots_url}')
        print(f'Humans with Emojis: {humans_emoji}, Bots with Emojis: {bots_emoji}')
        
        
        
                
