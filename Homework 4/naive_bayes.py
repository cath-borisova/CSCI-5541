"""
Katya Borisova (boris040) and Nick Truong (truon351)

****REPORT****
***Your results (precision, recall, and F-Measure):***
Recall: 0.78

Precision: 0.8297872340425532

F-Measure: 0.8041237113402062

***How you handled the tokens (i.e. what did you ignore, if anything?):***
We handle tokens by lowercasing everything in the text. In addition, we get rid
of all ' with an empty string "". Furthermore, we remove all punctuation except ! and ?
with a singular space " ". We also removed stopwords, did stemming with PorterStemmer,
and did negation-handling by appending "not_" to every word until the end of
the sentence after seeing a selection of negation words.

***What smoothing did you use?:***
We used Laplace smoothing.

***Did you add any other tricks (i.e. negation-handling, etc.)?:***
Other tricks we used were removing stopwords, did stemming with PorterStemmer,
and ignoring words that did not appear in training to avoid division by 0 errors.
We also utilized the technique of creating a bag of words, and we used
log() in our calculations to deal with underflow. For our training and dev 
set separation, we did an 80-20 split by having every 5th file going into
the development set and the rest going into the training set.
"""

"""
***HOW TO RUN***
What is needed in directory with naive_bayes.py to run:
Download the movie_reviews folder off of canvas then take out the pos
and neg folder in movie_reviews folder and put those folders in the same directoy
as the naive_bayes.py file. So, in one directory, there should be
naive_bayes.py, neg folder, pos folder, and movie_reviews folder.

How to run:
python3 naive_bayes.py
"""

import os
import nltk
from nltk.corpus import stopwords
import re
nltk.download('stopwords')
from nltk.stem import PorterStemmer
import math

class NaiveBayesModel: #class for training set
    def __init__(self):
        self.train_set = [] #all the words in the training set
        self.count = 0 #number of words in the training set
        self.word_freq = {} #frequency count of all the words in the training set


    def get_and_split_reviews(self, dir_name): #80-20 split
        dev_set = []
        index = 0
        for filename in os.listdir(dir_name):
            with open(os.path.join(dir_name, filename), 'r') as f: #open the file
                text = f.read() #read in the file
                if index % 5 == 0: #every 5th goes into the development set
                    dev_set.append(text.lower())
                else:
                    self.train_set.append(text.lower()) #rest go in the training set
                index +=1
        return dev_set


    def add_nots(self, words):
        new_words = []
        change = False
        for word in words:
            if word == "no" or word == "not":
                change = True
                new_words.append(word)
            elif change:
                new_word = "not_" + word
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words


    def create_bag_of_words(self):
        sentences = []
        for review in self.train_set: #go through every review
            sentences.extend(nltk.sent_tokenize(review)) #sentence segmentation
        tokenized_words = []
        for sentence in sentences:
            new_sentence = re.sub("[\']", "", sentence) #remove all ' (e.g. wouldn't --> wouldnt)
            new_sentence = re.sub("[^\w\s!\?]", " ", new_sentence) #remove all punctuation except ! and ?
            new_sentence = re.sub("_", " ", new_sentence) # remove all _
            words = nltk.word_tokenize(new_sentence) #word tokenization
            words = self.remove_stopwords(words)
            words = self.stem(words)
            words = self.add_nots(words) 
            tokenized_words.extend(words)
        self.train_set = tokenized_words


    def remove_stopwords(self, words):
        stop_words = set(stopwords.words('english'))
        new_words = []
        for word in words:
            if word not in stop_words:
                new_words.append(word)
        return new_words
        
    def stem(self, words):
        ps = PorterStemmer()
        new_words = []
        for word in words:
            new_word = ps.stem(word)
            new_words.append(new_word)
        return new_words

    def count_freq(self):
        self.count = len(self.train_set) #save the length of the training set
        for word in self.train_set:
            if word in self.word_freq:
                self.word_freq[word] += 1
            else:
                self.word_freq[word] = 1

        

class Development: #class for the development data
    def __init__(self, dev_set):
        self.dev = dev_set

    def add_nots(self, words):
        new_words = []
        change = False
        for word in words: #im playing around with what words should have nots added to:
            if word == "no" or word == "not":
                change = True
                new_words.append(word)
            elif change:
                new_word = "not_" + word
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words


    def tokenize(self):
        new_dev_set = []
        for review in self.dev: #go through every review
            new_review = []
            sentences = nltk.sent_tokenize(review) #sentence segmentation
            for sentence in sentences:
                new_sentence = re.sub("[\']", "", sentence) #remove all ' (e.g. wouldn't --> wouldnt)
                new_sentence = re.sub("[^\w\s!\?]", " ", new_sentence) #remove all punctuation except ! and ?
                new_sentence = re.sub("_", " ", new_sentence) # remove all _
                words = nltk.word_tokenize(new_sentence) #word tokenization
                words = self.remove_stopwords_and_stem(words)
                words = self.add_nots(words)
                new_review.extend(words) 
            new_dev_set.append(new_review)
        self.dev = new_dev_set
        
    def remove_stopwords_and_stem(self, words):
        stop_words = set(stopwords.words('english'))
        new_words = []
        ps = PorterStemmer()
        for word in words:
            if word not in stop_words:
                new_word = ps.stem(word)
                new_words.append(new_word)
        return new_words
    
     
        
            

def calc_liklihood(review, pos_word_freq, neg_word_freq, pos_train_len, neg_train_len, vocab):
    neg_prob = 0
    pos_prob = 0
    vocab_len = len(vocab)
    for word in review: #loops through every word in the review
        if word in vocab: #checks to see if the word appeared in either the positive or negative training set
            if word in pos_word_freq: 
                liklihood = math.log((pos_word_freq[word] + 1) / (pos_train_len + vocab_len), 2) #count(w, c) + 1/ sum of all words in c + len(V)
                pos_prob += liklihood
            else:
                liklihood = math.log(1 / (pos_train_len + vocab_len), 2) #count(w, c) + 1/ sum of all words in c + len(V) except count(w, c) = 0
                pos_prob += liklihood
            if word in neg_word_freq:
                liklihood = math.log((neg_word_freq[word] + 1) / (neg_train_len + vocab_len), 2) #count(w, c) + 1/ sum of all words in c + len(V)
                neg_prob += liklihood
            else:
                liklihood = math.log(1 / (neg_train_len + vocab_len), 2) #count(w, c) + 1/ sum of all words in c + len(V) except count(w, c) = 0
                neg_prob += liklihood
        #ignore words that did not appear in training
    return (pos_prob, neg_prob)

def is_pos(pos_liklihood, neg_liklihood, prior_pos, prior_neg): #returns True if the review is positive
    #for the positive class
    pos = pos_liklihood + prior_pos
    #for the negative class
    neg = neg_liklihood + prior_neg
    if pos > neg:
        return True
    else:
        return False

    
def test(dev, pos_word_freq, neg_word_freq, pos_train_len, neg_train_len, vocab, prior_pos, prior_neg, is_pos_class):
    correct = 0
    incorrect = 0
    for review in dev: #loops through every review
        liklihoods = calc_liklihood(review, pos_word_freq, neg_word_freq, pos_train_len, neg_train_len, vocab) #calculates the liklihood of the pos and neg class for the review
        pos_liklihood= liklihoods[0]
        neg_liklihood = liklihoods[1]
        if(is_pos(pos_liklihood, neg_liklihood, prior_pos, prior_neg) == is_pos_class): #checks to see if the classified class is the same as its actual class (is_pos_class)
            correct +=1
        else:
            incorrect += 1
    return (correct, incorrect) #returns how many classifications it got right and how many it got wrong

def recall(true_pos, false_neg):
    return true_pos/(true_pos + false_neg)

def precision(true_pos, false_pos):
    return true_pos / (true_pos + false_pos)

def f_measure(precision, recall):
    return (2*precision*recall)/ (precision + recall)

        

def main():
    """
    ***HOW TO RUN***
    What is needed in directory with naive_bayes.py to run:
    Download the movie_reviews folder off of canvas then take out the pos
    and neg folder in movie_reviews folder and put those folders in the same directoy
    as the naive_bayes.py file. So, in one directory, there should be
    naive_bayes.py, neg folder, pos folder, and movie_reviews folder.

    How to run:
    python3 naive_bayes.py
    """

    #creates a naive bayes model for the positive and negative class
    pos_class = NaiveBayesModel()
    neg_class = NaiveBayesModel()

    #creates a development set for the positive and negative reviews
    pos_dev_set = Development(pos_class.get_and_split_reviews("pos"))
    neg_dev_set = Development(neg_class.get_and_split_reviews("neg"))

    #calculated the prior probabilities
    prior_pos = math.log(len(pos_class.train_set)/ (len(neg_class.train_set) + len(pos_class.train_set)), 2) # number of documents in the positive class / number of all documents
    prior_neg = math.log(len(neg_class.train_set)/ (len(neg_class.train_set) + len(pos_class.train_set)), 2) # number of documents in the negative class / number of all documents

    #preprocessing of the training data
    pos_class.create_bag_of_words()
    neg_class.create_bag_of_words()

    pos_class.count_freq()
    neg_class.count_freq()

    #preprocessing of the development data
    pos_dev_set.tokenize()
    neg_dev_set.tokenize()

    #vocab is the set of all training words
    vocab = set(pos_class.train_set)
    vocab = vocab.union(neg_class.train_set)

    #classifies the pos development data (test returns how many it got right and how many it got wrong)
    pos_dev_classification = test(pos_dev_set.dev, pos_class.word_freq, neg_class.word_freq, pos_class.count, neg_class.count, vocab, prior_pos, prior_neg, True)
    true_positive = pos_dev_classification[0] #number of reviews in the positive class that were classified as positive (Correct)
    false_negative = pos_dev_classification[1] #number of reviews in the negative class that were classified as negative (Incorrect)
    neg_dev_classification = test(neg_dev_set.dev, pos_class.word_freq, neg_class.word_freq, pos_class.count, neg_class.count, vocab, prior_pos, prior_neg, False)
    true_negative = neg_dev_classification[0] #number of reviews in the negative class that were classified as negative (Correct) --we might not even need this value
    false_positive = neg_dev_classification[1] #number of reviews in the negative class that were classified as positive (Incorrect)

    # Calculating and printing the recall, precision, and F-measure scores. TP, FN, TN, and FP are also calcualted and printed out.
    r = recall(true_positive, false_negative)
    p = precision(true_positive, false_positive)
    f = f_measure(p, r)
    print("Recall: " + str(r) + "\n")
    print("Precision: " + str(p) + "\n")
    print("F-Measure: " + str(f) + "\n")


if __name__ == '__main__':
   main()