#Katya Borisova (boris040) and Mengzhen Li (li001618)
import nltk
import nltk.corpus
from nltk import pos_tag
from nltk import RegexpParser
from nltk import tree2conlltags
from pprint import pprint
import re
import string
#Dates are not named entities
nltk.download('treebank')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('conll2000')

#preprocess
def grammar(tags):
    gr = r"""
    NP:     
    {<THE>?<NNP>+<IN>(<NN>|<NNP>)<POS>(<NN>|<NNP>)+} 
    {<THE>?<NNP>+<IN>(<NN>|<NNP>)} 
    {<NNP><POS>(<NN>|<NNP>|<NNS>)+} 
    {<NNP>*<THE>?(<NNP>)*<NNP>(<NN>|<NNP>|<NNS>)*}
    {<NNP><NNS>?}
    {<JJ><NNP>+} 
    """
    parser = nltk.RegexpParser(gr)
    results = parser.parse(tags)
    iob_tagged = tree2conlltags(results)
    pprint(iob_tagged)


def preprocessing(text):
    edit_text = []
    for word in text:
        if '*' not in word and "'" not in word: #remove all the weird * tokens
            edit_text.append(word)
    edit_text = ' '.join(edit_text)
    print(edit_text)
    edit_text = re.sub('[^\w\s]', ' ', edit_text) #remove all punctuation
    edit_text = re.split(" ", edit_text)
    new_text = []
    for word in edit_text:
        if word not in string.punctuation:
            words = re.split("-", word) #split hyphenated words
            for w in words:
                new_text.append(w)  
    tags = pos_tag(new_text)
    new_tags = []
    dates = ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for tag in tags: #change tags to be <THE> for "the" or <DATE> for a month or day of the week (so that they are not considered NNP)
        if tag[0].lower() == 'the':
            new_tag = (tag[0], 'THE')
            new_tags.append(new_tag)
        else:
            new_tag = tag
            for date in dates:
                if re.match(date, tag[0]):
                    new_tag = (tag[0], "DATE")
                    break
            new_tags.append(new_tag)
    return new_tags



def main():
    wsj_text = nltk.corpus.treebank.sents()
    for i in range(4):
        tags = preprocessing(wsj_text[i])
        grammar(tags)

if __name__ == '__main__':
   main()