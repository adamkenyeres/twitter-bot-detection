import random
from sklearn.utils import shuffle

class DataAugmenter():
    def __init__(self, x, y, max_len, sequence_len):
        assert len(x) == len(y)
        self.accounts = x
        self.y = y
        self.max_len = max_len
        self.sequence_len = sequence_len
    
    def get_subset_from_index(self, account, length, start_index):
        subset = [account[i] for i in range(start_index, start_index + length)]
         #Pad the sequence
        if len(subset) < self.max_len:
            subset.extend([[0]*self.sequence_len]* (self.max_len - len(subset)))
        return subset
    
    def get_subset(self, account, length):
        assert len(account) >= length
        subsets = []
        lengths = []
        for start_index in range(0, len(account)-length + 1):
            lengths.append(length)
            subsets.append(self.get_subset_from_index(account, length, start_index))
        return subsets, lengths
    
    def shuffle(self, a, b, c):
        d = list(zip(a,b,c))
        random.shuffle(d)
        a,b,c = zip(*d)
        return list(a), list(b), list(c)
    
    def augment(self, sequence_lengths):
        new_accounts = []
        new_y = []
        sequence_length = []
        for length in sequence_lengths:
            print(f'Generating for {length}')
            for i in range(0, len(self.accounts)):
                subsets, lengths = self.get_subset(self.accounts[i], length)
                new_accounts.extend(subsets)
                new_y.extend([self.y[i]] * len(subsets))
                sequence_length.extend(lengths)
        print(sequence_length)
        #new_accounts, new_y, sequence_length =  shuffle(new_accounts, new_y, sequence_length, random_state=0)
        return self.shuffle(new_accounts, new_y, sequence_length)

a = [1,2,3]
b = [4,5,6]
c = [3,2,1]
d = list(zip(a,b,c))
random.shuffle(d)
a,b,c = zip(*d)
a
b
c
x = [['this is not s spam','Hi tweet', 'How are you','Hi'], [1,2,2,3]]
y = [0,1]
test = []
test.extend([0,0])
test.append(0)
test
x = [ [[1,2,32], [14,3,2], [10,20,30]],[[2,2,2], [3,3,3]] ]
augmenter = DataAugmenter(x,y, 4, 3)
accounts, y_aug, lengths = augmenter.augment([1, 2])
accounts
lengths
y_aug
#subset =  augmenter.get_subset([1,2,3,4,5], 3)
#subset
accounts
y_aug
random.randint(round(4*0.4), round(4*0.8))

for start_index in range(0, 3):
    print(start_index)
print(accounts)
random.randint(1,1)
