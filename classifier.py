import nltk
import random
from nltk.corpus import movie_reviews

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
    z.extractall(os.path.join(".", directory_label), members="A")

def main():

    download_zip(TRAINING_DATA, "training_set")
    download_zip(TRAINING_DATA_CURRENT, "training_set_current")


if __name__=="__main__":
    main()