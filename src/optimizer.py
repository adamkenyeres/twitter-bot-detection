from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from lstm import TweetClassifier, AccountClassifier
from ray.tune import CLIReporter
from ray.tune.schedulers import AsyncHyperBandScheduler

def create_account_classifier(space, x_train, y_train, x_val, y_val, lengths, epochs=10):
    name = 'Tweet_Classifier_' + str(space['batch_size']) +'_' + str(space['lstm_layers'])  +'_' + str(space['lr']) +'_' + str(space['lstm_hidden_size'])
    account_classifier = AccountClassifier(input_size=space['input_size'],
                                      batch_size = space['batch_size'],
                                      lstm_layers = space['lstm_layers'],
                                      lr=space['lr'],
                                      lstm_hidden_size=space['lstm_hidden_size'],
                                      bidirectional = True,
                                      weight_decay = space['weight_decay'],
                                      lstm_dropout = space['lstm_dropout'],
                                      fc_drouput = space['fc_drouput'],
                                      nn_size = space['nn_layers'], 
                                      optimize = True)
                
    account_classifier.train_model(x_train, y_train, x_val, lengths, y_val, epochs=20)


def create_model(space, embedding, x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val, epochs=10):
    name = 'Accoutn_Classifier_' + str(space['batch_size']) +'_' + str(space['lstm_layers'])  +'_' + str(space['lr']) +'_' + str(space['lstm_hidden_size'])
    tweet_classifier = TweetClassifier(input_size=1000,
                                      embedding_dim = 50,
                                      batch_size = space['batch_size'],
                                      lstm_layers = space['lstm_layers'],
                                      preprocessor=0,
                                      lr=space['lr'],
                                      lstm_hidden_size=space['lstm_hidden_size'],
                                      pretrained_model = embedding,
                                      bidirectional = True,
                                      weight_decay = space['weight_decay'],
                                      lstm_dropout = space['lstm_dropout'],
                                      fc_drouput = space['fc_drouput'],
                                      nn_size = space['nn_layers'], 
                                      optimize = True,
                                      name=name)
                
    tweet_classifier.train_model(x_tweets_train, y_tweets_train, x_tweets_val, y_tweets_val, epochs=epochs)
    
    
class Optimizer():
    def __init__(self, x_train, y_train, x_val,
                 y_val, embedding, 
                 preprocessor, vocab = 10000,
                 embedding_dim=50, epochs=20):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.embedding_dim = embedding_dim
        self.preprocessor = preprocessor
        self.epochs = epochs
        self.vocab = vocab
        self.embedding = embedding
        
    def optimize(self, space, current_best_params, trials, lengths=0, optimizeTweetClassifier=True):
        search_alg = HyperOptSearch(
            space,
            metric="val_acc",
            mode="max",
            points_to_evaluate = current_best_params)
        
        scheduler = AsyncHyperBandScheduler(
              time_attr='iterations',
              metric="val_acc",
              mode="max",
              reduction_factor=2,
              brackets=1,
              grace_period=5)

        if optimizeTweetClassifier:
            analysis = tune.run(
                tune.with_parameters(create_model, 
                                     embedding=self.embedding,
                                     x_tweets_train=self.x_tweets_train,
                                     y_tweets_train = self.y_tweets_train, 
                                     x_tweets_val = self.x_tweets_val,
                                     y_tweets_val = self.y_tweets_val, 
                                     epochs=self.epochs),
                search_alg=search_alg, 
                scheduler=scheduler,
                num_samples=trials,
                resources_per_trial={"cpu": 2, "gpu": 1}
                )
        else:
            analysis = tune.run(
                tune.with_parameters(create_account_classifier, 
                                     x_train=self.x_train,
                                     y_train = self.y_train, 
                                     x_val = self.x_val,
                                     y_val = self.y_val,
                                     lengths=lengths,
                                     epochs=self.epochs),
                search_alg=search_alg, 
                scheduler=scheduler,
                num_samples=trials,
                resources_per_trial={"cpu": 2, "gpu": 1}
                )
            

        print("Best config: ", analysis.get_best_config(
            metric="val_acc", mode="max"))
        
        # Get a dataframe for analyzing trial results.
        df = analysis.trial_dataframes
        return analysis, df
          #ray.shutdown()