import nltk
import random
import os

from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize, sent_tokenize


TRAINING_DATA = "http://users.csc.calpoly.edu/~foaad/proj1F21_files.zip"
TRAINING_DATA_CURRENT = "http://users.csc.calpoly.edu/~foaad/proj1S23_files.zip"

def find_features(document, word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def download_zip(url, directory_label):
    import requests, zipfile, io, os
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(os.path.join(".", directory_label))

def main():
    from bs4 import BeautifulSoup
    def get_dir_tokens(directory, filter=""):
        import codecs
        tokens=[]
        print(directory)
        for html in os.listdir(directory):
            if filter in html:
                with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
                    raw = BeautifulSoup(file, 'html.parser').get_text()
                    tokens += word_tokenize(raw)
        return tokens


    download_zip(TRAINING_DATA, "training_set")
    download_zip(TRAINING_DATA_CURRENT, "training_set_current")
    training_tokens_A = get_dir_tokens(os.path.join(".", os.path.join("training_set", "proj1F21_files")), "A")
    training_tokens_B = get_dir_tokens(os.path.join(".", os.path.join("training_set", "proj1F21_files")), "B")
    new_training_tokens_A = get_dir_tokens(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), "A")
    new_training_tokens_B = get_dir_tokens(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), "B")
    print(training_tokens_A[100])
    print(training_tokens_B[100])
    print(new_training_tokens_A[100])
    print(new_training_tokens_B[100])

if __name__=="__main__":
    main()