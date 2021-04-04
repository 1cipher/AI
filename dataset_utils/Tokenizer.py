from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import re
from nltk import word_tokenize

class Tokenizer(object):
    def __init__(self):
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))             #Core function for tokenization
    def __call__(self, doc):
        stemmer = PorterStemmer()
        clean = [self.regex.sub('',w) for w in word_tokenize(doc)]
        return [stemmer.stem(t) for t in clean if( len(t)>2 and t not in stopwords.words('english'))]

