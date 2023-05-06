import re
import sys
# Katya Borisova (boris040) and Nick Truong (truon351)


def word_tokenize(text): # word_tokenize() considers punctuation as words. Also, kept words like "__hello__" or "_hi_" or "can't" or "hasn't" or "mr." or "mrs." together with their punctuation.
    words = re.split(r"\s", text) # split on all spaces, so a list of all words with and without punctuation attached is created.
    new_words = [] # This list will contain all words and punctuation separated.
    
    for word in words: # loop through all words to find which words have punctuation attached to separate the punctuation from the word.
        low_word = word.lower()

        match_attached_abrev = re.match(r"\w*(\‘|\”|\“|\’|\—|\-|\'|\"|\(|\)|\[|\]|\_|\{|\}|\.|\?|\!|\:|\;)+(mr\.|mrs\.|dr\.)", low_word)
        """
        The match above matches with the abbreviations "mr.", "mrs.", or "dr." with 1 or more punctuations attached
        before them and possibly 0 or more characters attached before those punctuations. Examples of matches are:
        "(mr.", "“mrs.", "situation.-mr.", or "hello-mr."
        """
        if match_attached_abrev != None:
            split_punc = re.split(r"(\‘|\”|\“|\’|\—|\-|\'|\"|\(|\)|\[|\]|\_|\{|\}|\.|\?|\!|\:|\;)(?!$)", low_word)
            """
            So, this split above splits on any of those listed punctuations above if those punctuations are not at
            the end of the sentence. This is to avoid the "." in the abbreviations since the "." in the
            abbreviations is the only punctuation that will be at the end of the string.
            """
            new_words.extend(split_punc)
            continue

        match_quot_contractions = re.match(r"[\’\'\‘]*\w+[\’\'\‘]\w+[\’\'\‘]*", low_word)
        """
        The match above matches contraction words with 0 or more single quotes after or before attached
        to the contraction. Examples of matches are: "can't", "'can't", "can't'", "'can't'"
        """
        if match_quot_contractions != None:
            split_punc = re.split(r"(^\'|\'$|^\’|\’$|^\‘|\‘$)", low_word)
            """
            The only single quotes that we want to split on are at the end or start of the string since
            we do not want to split on the single quote that makes a contraction which is in-between 2 characters.
            """
            new_words.extend(split_punc)
            continue

        match_other_contractions = re.match(r"\w*(\‘|\”|\“|\’|\—|\-|\'|\"|\(|\)|\[|\]|\_|\{|\}|\:|\;)*\w+[\’\'\‘]\w+(\‘|\”|\“|\’|\—|\-|\'|\"|\(|\)|\[|\]|\_|\{|\}|\:|\;)*", low_word)
        """
        The match above matches with a pattern of:
        (0 or more characters) + (0 or more punctuation) + contraction + (0 or more punctuation)
        Examples of matches are: "-can't", "stop-don't.", "-hasn't:", and "can't:"
        This case is needed to deal with the other punctuations other than single quotes that might be attached
        to the contractions.
        """
        if match_other_contractions != None:
            split_punc = re.split(r"([^\'\’\‘\w])", low_word)
            """
            This split above splits on everything but ', ’, ‘, and characters.
            This is because the case of single quotes has already been dealt with.
            """
            new_words.extend(split_punc)
            continue

        match_abrev = re.match("dr\.|etc\.|mr\.|mrs\.|dept\.|est\.|al\.|i\.e\.|u\.s\.|u\.s\.a\.|inc\.|jr\.|ltd\.|sr\.|vs\.|cal\.|cm\.|oz\.|ft\.|gal\.|hr\.|in\.|kg\.|km\.|m\.|mg\.|min\.|mm\.|sec\.|sq\.|vol\.|approx\.|appt\.|apt\.|a\.s\.a\.p\.|d\.i\.y\.|e\.t\.a\.|misc\.|ms\.|no\.|r\.s\.v\.p\.|tel\.|temp\.|vet\.|jan\.|feb\.|aug\.|sept\.|oct\.|nov\.|dec\.|gov\.|govt\.|ave\.|b\.a\.|b\.s\.", low_word)
        # The match above matches with any abbreviations manually listed. For example: "mr.", "u.s.", "jr.", "sr."
        match_punc_forward = re.match(r"[\w\d]+[\‘\”\“\’\—\-\#\'\"\,\;\:\-\(\)\[\]\_\{\}\.\?\!\%\:\;]+", low_word)
        # The match above matches with any word with punctuation attached after it. For example: "hello.", "hi!", "stop)"
        match_punc_backward = re.match(r"[\‘\”\“\’\—\-\'\"\,\;\:\-\(\)\[\]\_\{\}\.\?\!\#\%\:\;]+[\w\d]+", low_word)
        # The match above matches with any word with punctuation attached before it. For example: "(hello", "'hi", "-stop"
        if (match_abrev == None) and (match_punc_forward != None or match_punc_backward != None):
            # match_abbrev != None means that there is no splitting that needs to be done since it is just the abbreviation with a "."
            split_punc = re.split(r"(\W)", low_word)
            # This split above splits on all punctuation.
            new_words.extend(split_punc)
            continue

        new_words.append(low_word) # any word without punctuation needed to separated from the word is appended to the list here.

    new_words = list(filter(None, new_words)) # filter to get rid of empty strings.
    return new_words





def sent_tokenize(text): # splits on 1 or more spaces, —, or - between sentences.
    new_text = re.sub("\n\s*", " ", text).lower() # subs all the newlines with 0 or more spaces in the text with a single space " ".
    sentences = re.split(r"(?<!.{3}dr\.|.{2}u\.s\.|u\.s\.a\.|.{2}etc\.|.{3}mr\.|.{2}mrs\.|.{1}dept\.|.{2}inc\.|.{3}jr\.|.{2}ltd\.|.{3}sr\.|.{3}vs\.|.{3}cm\.|.{3}oz\.|.{3}ft\.|.{3}hr\.|.{3}kg\.|.{3}km\.|.{3}mg\.|.{3}mm\.|.{2}sec\.|.{3}sq\.|.{2}vol\.|.{1}appt\.|.{2}apt\.|.{2}asap|.{2}diy\.|.{1}misc\.|.{1}rsvp\.|.{1}temp\.|.{2}jan\.|.{2}feb\.|.{2}aug\.|.{1}sept\.|.{2}oct\.|.{2}nov\.|.{2}dec\.|.{2}gov\.|.{1}govt\.)(?<=.{3}\.|.{2}\.\"|.{3}\!|.{3}\?|.{2}\?\"|.{2}\!\"|.{2}\.\”|.{2}\?\”|.{2}\!\”|.{1}\.\.\.|\.\.\.\"|\.\.\.\'|\.\.\.\’|\.\.\.\”|.{2}\?\'|.{2}\!\'|.{2}\.\'|.{2}\?\’|.{2}\!\’|.{2}\.\’|.{2}\?\‘|.{2}\!\‘|.{2}\.\‘|\.\.\.\‘|.{2}\?\)|.{2}\!\)|.{2}\.\)|\.\.\.\)|.{2}\?\:|.{2}\!\:|.{2}\.\:|\.\.\.\:|.{2}\?\;|.{2}\!\;|.{2}\.\;|\.\.\.\;)[\s\—\-]+", new_text)
    """
    The main idea of this split is to split on 1 or more spaces, —, or - that
    before it has punctuation that typically ends a sentence. Also, there is a 
    negative lookbehind to make sure that it is not mistaking a "." with a "."
    from an abbreviation. Examples of splits created would be: "hello...", "hey there stop!"
    Furthermore, the reason dashes are included with spaces is because in the text-files, sometimes
    dashes are treated as spaces leading to the next sentence after a "."
    """
    return list(filter(None, sentences)) # filter to get rid of empty strings



"""
How to run:
python3 tokenizer.py <file_name>.txt
This runs both word_tokenize and sent_tokenize.
"""
def main():
    file = open(sys.argv[1], encoding="utf8")
    whole_string = file.read()
    print("this is word_tokenize:")
    print(word_tokenize(whole_string))
    print("this is sent_tokenize:")
    print(sent_tokenize(whole_string))
    file.close()
    


if __name__ == '__main__':
   main()
