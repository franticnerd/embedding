from nltk.corpus import stopwords
from textblob import Word
import re

class TextProcessor:

    def __init__(self, min_word_length=3, stem=True, stopword_file=None):
        self.stem = stem  # stem or not
        self.min_word_length = min_word_length # minimum word length
        self.stopwords = set(stopwords.words('english'))
        self.word_pattern = re.compile(r'[0-9a-zA-Z]+')
        if stopword_file is not None:
            self.load_stopwords(stopword_file)

    '''
    Allow a user to specify more stopwords through a file;
    Each line in the file is a stop word.
    '''

    def load_stopwords(self, stopwords_file):
        with open(stopwords_file, 'r') as fin:
            for line in fin:
                stopword = line.strip()
                self.stopwords.add(stopword)

    '''
    Parse a string into a list of words.
    Perform
    '''
    def parse_words(self, input_string):
        tokens = re.findall(self.word_pattern, input_string)
        words = []
        for token in tokens:
            word = self.clean(token)
            if word is not None:
                words.append(word)
        return words

    # clean one word.
    def clean(self, token):
        #  Check whether the word is too short
        if(len(token) < self.min_word_length):
            return None
        word = Word(token.lower()).lemmatize()
        #  Check whether the word is a stop word
        if len(word) < self.min_word_length or word in self.stopwords:
            return None
        else:
            return word

if __name__ == '__main__':
    word_processor = TextProcessor(min_word_length = 3)
    s = 'hello, 13423, This is@ went octopi just a test for 12you!. Try it http://'
    print word_processor.parse_words(s)
