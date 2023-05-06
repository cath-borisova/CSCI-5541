import sys
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk import FreqDist
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re
import string

def remove_stopwords(words):
    stop_words = stopwords.words("english")
    more_stop_words = ["i", "\n", "it"]
    stop_words.extend(more_stop_words)
    new_words = []
    for word in words:
        if word not in stop_words:
            new_words.append(word)
    return new_words

def get_words(f):
    words = []
    for line in f:
        line = re.sub('[^\w\s]', '', line)
        words.extend(word_tokenize(line))
    return words

def convert_to_lowercase(words):
    new_words = []
    for word in words:
        new_words.append(word.lower())
    return new_words

def get_frequency(words, maxLength):
    frequency = FreqDist(words)
    for freq in frequency.most_common(maxLength):
        print(freq[0])


def main():
    f = open(sys.argv[1])
    words = get_words(f)
    words = convert_to_lowercase(words)
    words = remove_stopwords(words)
    get_frequency(words, 5)
    f.close()

if __name__ == '__main__':
   main()