# import nltk
import random
import os
import codecs
import re
from statistics import variance
from bs4 import BeautifulSoup
# from nltk.corpus import movie_reviews
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer as ltz
from nltk.classify import DecisionTreeClassifier
from nltk.tag import pos_tag

TRAINING_DATA = "http://users.csc.calpoly.edu/~foaad/proj1F21_files.zip"
TRAINING_DATA_CURRENT = "http://users.csc.calpoly.edu/~foaad/proj1S23_files.zip"
PATH_SIMILARITY_CUTOFF = 0.24
TEST_SIZE = 0.2

# features for both classifiers
#   average sentence length
#   average word length 
#   number of sentences
#   number of words

# topic classifier features:
#   number of proper nouns (or just names)
#   words with semantic path length <= 3 to vehicle (particularly car)
#   words with semantic path length <= 3 to person (or admire)

# writer classifier:
#   punctuation per sentence
#   number of appearences of (am are is was were be being been) per sentence
#   sentence length variance
#   word length variance
#   

# given a list of sentences return the
# average sentence length
def get_average_sentence_length(sents):
    total_length = 0
    for sent in sents:
        total_length += len(sent)
    return float(total_length) / len(sents)

# given a list of words, return the average
# word length
def get_average_word_length(words):
    total_length = 0
    for word in words:
        total_length += len(word)
    return float(total_length) / len(words)

# given a list of words, return the
# number of proper nouns it contains
def get_num_NNP(words):
    count = 0
    tagged = pos_tag(words)
    for word_tag in tagged:
        if word_tag[1] == "NNP":
            count += 1
    return count

# given a list of words, return the 
# number of words sufficiently similar
# to the target
def get_sim_count(words, target):
    count = 0
    target_syn = wn.synsets(target)[0]
    for word in words:
        syns = wn.synsets(word)
        if len(syns) > 0:
            word_syn = wn.synsets(word)[0]
            if word_syn.path_similarity(target_syn) > PATH_SIMILARITY_CUTOFF:
                count += 1
    return count

# given a list of sentences, return 
# the number of punctuation marks
# per sentence
def get_punc_density(sents):
    count = 0
    for sent in sents:
        tokens = word_tokenize(sent)
        tagged = pos_tag(tokens)
        for word_tag in tagged:
            if word_tag[1] == ".":
                count += 1
    return float(count) / len(sents)

# given a list of sentences, return 
# the average number of occuences of 
# am are is was were be being or been
# per sentence
def get_tobe_count(sents):
    count = 0
    be_words = set(["am", "are", "is", "was", "were", "be", "being", "been"])
    for sent in sents:
        for word in sent:
            if word in be_words:
                count += 1
    return float(count) / len(sents)

# given a list of sentences, return
# the variance of their length
def get_sent_variance(sents):
    sent_lengths = []
    for sent in sents:
        sent_lengths.append(len(sent))
    return variance(sent_lengths)

# given a list of words, return
# the variance of their length
def get_word_variance(words):
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    return variance(word_lengths)

def get_bag_o_words(words, stopwords):
    lemmatizer = ltz()
    freqs = {}
    for word in words:
        formatted = lemmatizer.lemmatize(word.lower())
        if formatted not in stopwords:
            if formatted not in freqs:
                freqs[formatted] = 1
            else:
                freqs[formatted] += 1
    return freqs

def get_bigrams(words):
    bgrams = bigrams([word.lower() for word in words])
    freqs = {}
    for gram in bgrams:
        if gram in freqs:
            freqs[gram] += 1
        else:
            freqs[gram] = 1
    return freqs


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

# return a list of words for all html documents in 
# the given directory 
def get_dir_tokens(directory, filter=""):
    tokens=[]
    print(directory)
    for html in os.listdir(directory):
        if filter in html:
            with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
                raw = BeautifulSoup(file, 'html.parser').get_text()
                tokens += word_tokenize(raw)
    return tokens
    
# get a list of sentences in the file 
# at the given path
def get_file_sents(path):
    sents=[]
    with open(path, "r") as file:
        raw = BeautifulSoup(file, 'html.parser').get_text()
        sents = sent_tokenize(raw)
    return sents

# get a list of words in the file 
# at the given path
def get_file_words(path):
    words=[]
    with open(path, "r") as file:
        raw = BeautifulSoup(file, 'html.parser').get_text()
        words = word_tokenize(raw)
    return words

def print_file(path):
    with open(path, "r") as file:
        raw = BeautifulSoup(file, 'html.parser').get_text()
        print(raw)

# get all training data from a directory
def get_features(directory, filter="", topic=True):
    data = []
    for html in os.listdir(directory):
        if filter in html:
            with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
                raw = BeautifulSoup(file, 'html.parser').get_text()
                sents = sent_tokenize(raw)
                words = word_tokenize(raw)
                sw = set(stopwords.words('english'))
                #both classifiers
                features = {}
                features["sent_len"] = get_average_sentence_length(sents)
                features["word_len"] = get_average_word_length(sents)
                features["num_sents"] = len(sents)
                features["num_words"] = len(words)
                bow = get_bag_o_words(words, sw)
                bgrams = get_bigrams(words)
                features.update(bow)
                features.update(bgrams)
                if topic:
                    features["NNP"] = get_num_NNP(words)
                    features["vehicle_dist"] = get_sim_count(words, "vehicle")
                    features["person_dist"] = get_sim_count(words, "person")
                    if "A" in html:
                        data.append((features, "A"))
                    else:
                        data.append((features, "B"))
                else:
                    features["punct_dens"] = get_punc_density(sents)
                    features["tobe"] = get_tobe_count(sents)
                    features["sent_var"] = get_sent_variance(sents)
                    features["word_var"] = get_word_variance(words)
                    found = re.search("\d\d\d\d", html)
                    data.append((features, found[0]))              
    return data

def train_test_bayes(data):
    split = int(len(data) * TEST_SIZE)
    random.shuffle(data)
    train_set = data[split:]
    test_set = data[:split]
    test_features = [x[0] for x in test_set]
    test_categories = [x[1] for x in test_set]

    classifier = NaiveBayesClassifier.train(train_set)
    # classifier = DecisionTreeClassifier.train(train_set)
    accuracy = get_accuracy(test_features, test_categories, classifier)
    print(accuracy)

def get_accuracy(test_features, test_categories, classifier):
    correct = 0
    guesses = classifier.classify_many(test_features)
    for i in range(len(guesses)):
        if test_categories[i] == guesses[i]:
            correct += 1
    return float(correct) / len(test_features)

def main():
    download_zip(TRAINING_DATA, "training_set")
    download_zip(TRAINING_DATA_CURRENT, "training_set_current")

    # data = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")))
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")))
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")))
    data = data_1 + data_2
    train_test_bayes(data)

    # training_tokens_A = get_dir_tokens(os.path.join(".", os.path.join("training_set", "proj1F21_files")), "A")
    # training_tokens_B = get_dir_tokens(os.path.join(".", os.path.join("training_set", "proj1F21_files")), "B")
    # new_training_tokens_A = get_dir_tokens(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), "A")
    # new_training_tokens_B = get_dir_tokens(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), "B")
    # print(training_tokens_A[100])
    # print(training_tokens_B[100])
    # print(new_training_tokens_A[100])
    # print(new_training_tokens_B[100])
    # print_file("./training_set/proj1F21_files/proj1F21_1483_A.html")

if __name__=="__main__":
    main()