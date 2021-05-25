from dataset_loader import DataSetLoader
import nlpaug.augmenter.word as naw
from preprocessor import Preprocessor

class TweetAugmenter():
    def __init__(self, embedding, glove_path, x = 0, y = 0, hash = 0, truth = 0, path=0):
        self.embedding = embedding
        self.path = path
        self.truth = truth
        self.glove_path = glove_path
        self.dataset_loader = DataSetLoader(self.path)
        self.preprocessor = Preprocessor()
        self.x = x
        self.y = y
        self.hashes = hash
        
        #self.aug_syn = naw.SynonymAug(aug_src='wordnet')
        self.aug_glove = naw.WordEmbsAug(
            model_type='glove', model_path=self.glove_path,
            action="substitute")
    
    def get_data(self):
        if self.x == 0:
            df, humans, bots = self.dataset_loader.get_df_humans_bots(truth=self.truth, should_preprocess=False)
            x, y, hashes = self.dataset_loader.get_x_y(humans, bots)
            return x, y, hashes
        else:
            return self.x, self.y, self.hashes
    
    def augment_text(self, text):
        #augmented_text = self.aug_syn.augment(text)
        augmented_text = self.aug_glove.augment(text)
        prep_text = self.preprocessor.replace_special_chars(augmented_text)
        prep_text = self.preprocessor.remove_space_before_special_char(prep_text)
        return prep_text
        
    def augment(self):
        accounts, y, hashes = self.get_data()
        y = y
        augmented_accounts = []
        index = 0
        size = len(accounts)
        for account in accounts:
            augmented_account = []
            proccessed_percante = index/size*100
            if index%10 == 0:
                print(f'Augmented {proccessed_percante}% of accounts')
            for tweet in account:
                augmented_account.append(self.augment_text(tweet))
            augmented_accounts.append(augmented_account)
            index += 1
        
        return augmented_accounts, y, hashes, accounts
