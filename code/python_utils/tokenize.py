import spacy
import nltk

class Tokenizers:
    def __init__(self,nltk_dir=None,spacy_model="en_core_web_sm"):
        nltk.download('stopwords',download_dir=nltk_dir)
        from nltk.corpus import stopwords
        self.stp=set(stopwords.words("english"))
        self.nlp=spacy.load(spacy_model,disable=["tagger", "parser","ner"])  #Only Tokennize

    def nlp_tokenize(self,string):
        return [token.text for token in self.nlp(string)]

    def nlp_tokenize_lower(self,string):
        return [token.text.lower() for token in self.nlp(string) if token.text.lower() not in self.stp]

    def nlp_tokenize_entity(self,string):
        return [token.text.lower() for token in self.nlp(string) if token.text.lower() not in self.stp and (not token.is_punct)]

    def split_tokenizer(self,string):
        return string.split(" ")