
import re
from collections import Counter

f = open("./data/en_train.sample", "r")
data=[]
for x in f:
  data.append(x)

max_len=0
for x in data:
    n=len(x.split(' '))
    if n>max_len:
        max_len=n

print(max_len)



def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

def generate_vocabulary(train_captions, min_threshold):
    """
    Return {token: index} for all train tokens (words) that occur min_threshold times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    """  
    #convert the list of whole captions to one string
    concat_str = ' '.join([str(elem).strip('\n') for elem in train_captions]) 
    #divide the string tokens (individual words), by calling the split_sentence function 
    individual_words = split_sentence(concat_str)
    #create a list of words that happen min_threshold times or more in that string  
    condition_keys = sorted([key for key, value in Counter(individual_words).items() if value >= min_threshold])
    #generate the vocabulary(dictionary)
    result = dict(zip(condition_keys, range(len(condition_keys))))
    return result

# train_captions = ['Nory was a Catholic because her mother was a Catholic, and Nory’s mother was a Catholic because her father was a Catholic, and her father was a Catholic because his mother was a Catholic, or had been.',
#                   'I felt happy because I saw the others were happy and because I knew I should feel happy, but I wasn’t really happy.',
#                   'Almost nothing was more annoying than having our wasted time wasted on something not worth wasting it on.']
train_captions=data
vocab=generate_vocabulary(train_captions, min_threshold=5)
print(len(vocab.keys()))