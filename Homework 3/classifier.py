#Katya Borisova (boris040) and Nick Truong (truon351)
#Report:

#Encoding: utf8

#Information in Language Model: everygrams that go up to bigrams

#Smoothing method: Add one smoothing

#Other tweaks: Removed all apostrophes and quotations because we 
#found there were too many kinds of them that were appearing in 
#our text - plus the sentences could still flow without them

#Results:
# austen  91.15504682622269%
# dickens 74.54728370221329%
# tolstoy 78.62318840579711%
# wilde   72.33848953594176%

from cmath import inf
import re
import sys
import nltk
nltk.download('punkt')
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.preprocessing import padded_everygrams
from nltk.lm import Laplace
from nltk.corpus import stopwords
nltk.download('stopwords')

#class for when the program is run without a test flag
class Dev:
    def __init__(self, authorlist):
        self.authorlist = authorlist.split("\n") #creates a list of the filenames in authorlist

    def split_data(self, text): #splits the data into training and development with an 80-20 split
        train_data = [] 
        dev_data = []
        index = 0
        while index < len(text):
            if index % 5 == 0: #every 5th sentence goes into the development set, all other sentences go into the training set
                dev_data.append(text[index])
            else:
                train_data.append(text[index])
            index += 1
        return (train_data, dev_data) 

    def edit_text_dev(self, text): #preprocessing for the developmental set
        dev = []
        for sentence in text:
            new_sentence = re.sub("[\‘\”\“\’\'\"]", "", sentence) #remove all quotation marks and apostrophes 
            new_sentence = nltk.word_tokenize(new_sentence) #tokenize into words
            new_sentence = list(padded_everygrams(2, new_sentence)) #convert the sentence into a padded everygram that goes up to bigrams
            dev.append(new_sentence)
        return dev

    def edit_text_train(self, text): #preprocessing for the training set
        new_text = []
        for sentence in text:
            new_sentence = re.sub("[\‘\”\“\’\'\"]", "", sentence) #remove all quotation marks and apostrophes
            new_sentence = nltk.word_tokenize(new_sentence) #tokenize into words
            new_text.append(new_sentence)
        train, vocab = padded_everygram_pipeline(2, new_text) #convert the text into a padded everygram that goes up to bigrams to create the training set and vocabulary
        return (train, vocab)

    def test(self, models, dev_sets):
        for i in range(len(dev_sets)): #test every development set
            count_correct_author = 0
            correct_author = models[i][0] 
            for sentence in dev_sets[i]: #run through every sentence in the current development set
                new_sentence = list(sentence)
                author = None
                perplexity = inf
                for model in models: #run the current sentence through every model
                    p = model[1].perplexity(new_sentence) #calculate the perplexity of the sentence with the model
                    if p < perplexity: 
                        perplexity = p #set the perplexity to the lowest calculated one
                        author = model[0] #set the author to the classified author (model with lowest perplexity)
                if author == correct_author: #count how many times the sentence was classified with the correct author
                    count_correct_author +=1
            accuracy = count_correct_author / len(dev_sets[i])
            accuracy *= 100 
            print(correct_author + "\t" + str(accuracy) + "%") #return accuracy

    def run(self):
        print("splitting into training and development... ")
        print("training LMs... (this may take a while)")
        models = [] #[(author's name, model)...] - array of tuples that hold the author's name and the model
        dev_sets = []
        for author in self.authorlist:
            #getting the author's name
            name = author.replace(".txt", "")
            name = name.replace("_utf8", "")
            text = open(author, encoding="utf8").read() #opening the author's textfile
            #Preprocessing
            text = text.lower() #convert the text to lowercase
            text = nltk.sent_tokenize(text) #sentence segmentation
            split = self.split_data(text) #split the data into development and training - the function returns (training data, development data)
            train_data = split[0]
            dev_data = split[1]
            train = self.edit_text_train(train_data) #preprocess the training set - the function returns (training data, vocab)
            train_data = train[0]
            vocab = train[1]
            dev_data = self.edit_text_dev(dev_data) #preprocess the development set
            #Training
            model = Laplace(2) 
            model.fit(train_data, vocab) 
            models.append((name, model))
            dev_sets.append(dev_data)
        #Testing
        print("Results on dev set:")
        self.test(models, dev_sets)

#class for when the program is run with the test flag
class Test:
    def __init__(self, authorlist, test):
        self.authorlist = authorlist.split("\n") #creates a list of the filenames in authorlist
        self.testfile = test #file to test

    def edit_text_test(self, text): #preprocessing for the test set
        data = []
        text = nltk.sent_tokenize(text) #sentence segmentation
        for sentence in text:
            new_sentence = re.sub("[\‘\”\“\’\'\"]", "", sentence) #remove all quotation marks and apostrophes
            new_sentence = nltk.word_tokenize(new_sentence) #tokenize into words
            new_sentence = list(padded_everygrams(2, new_sentence)) #convert the sentence into a padded everygram that goes up to bigrams
            data.append(new_sentence)
        return data
    
    def edit_text_train(self, text): #preprocessing for the training set
        new_text = []
        text = nltk.sent_tokenize(text) #sentence segmentation
        for sentence in text:
            new_sentence = re.sub("[\‘\”\“\’\'\"]", "", sentence) #remove all quotation marks and apostrophes
            new_sentence = nltk.word_tokenize(new_sentence) #tokenize words
            new_text.append(new_sentence)
        data, vocab = padded_everygram_pipeline(2, new_text) #convert the sentence into a padded everygram that goes up to bigrams
        return (data, vocab) #return training data and vocab
    
    def test(self, models, test_data):
        correct_author = 0
        for sentence in test_data: #loop through every sentence in the test data
            new_sentence = list(sentence)
            author = None
            perplexity = inf
            for model in models: #run the current sentence through every model
                p = model[1].perplexity(new_sentence) #calculate the perplexity of the sentence with the model
                if p < perplexity:
                    perplexity = p #set the perplexity to the lowest calculated one
                    author = model[0] #set the author to the classified author (model with lowest perplexity)
            if author == "austen":
                correct_author +=1
            print(author) #print the classified data
        print(correct_author/len(test_data) * 100)
            

    def run(self):
        print("training LMs... (this may take a while)")
        models = [] #[(author's name, model)...] - array of tuples that hold the author's name and the model
        for author in self.authorlist:
            #getting author's name
            name = author.replace(".txt", "")
            name = name.replace("_utf8", "")
            text = open(author, encoding="utf8").read() #opening the author's textfile
            #Preprocessing
            text = text.lower() #convert the text to lowercase
            train = self.edit_text_train(text) #preprocess the training set - the function returns (training data, vocab)
            train_data = train[0]
            vocab = train[1]
            #Training
            model = Laplace(2)
            model.fit(train_data, vocab)
            models.append((name, model))
        #Testing
        test_data = self.testfile.lower() #convert the text to lowercase
        test_data = self.edit_text_test(test_data) #preprocess the test set
        self.test(models, test_data)


def main():
    if len(sys.argv) == 2: #if run without test flag
        authorlist = open(sys.argv[1], encoding="utf8")
        model = Dev(authorlist.read())
        model.run()
        authorlist.close()
    else: #if run with test flag
        authorlist = open(sys.argv[1], encoding="utf8")
        test = open(sys.argv[3], encoding="utf8")
        model = Test(authorlist.read(), test.read())
        model.run()
        authorlist.close()
        test.close()

if __name__ == '__main__':
   main()