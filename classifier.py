
import random
import os
import codecs
import re
from enum import Enum
from statistics import variance
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.stem import WordNetLemmatizer as ltz
from nltk.classify import DecisionTreeClassifier
from nltk.tag import pos_tag
from sklearn.metrics import precision_recall_fscore_support

TRAINING_DATA = "http://users.csc.calpoly.edu/~foaad/proj1F21_files.zip"
TRAINING_DATA_CURRENT = "http://users.csc.calpoly.edu/~foaad/proj1S23_files.zip"
PATH_SIMILARITY_CUTOFF = 0.24
TEST_SIZE = 0.2

TOGGLE = False

class FType(Enum):
    DOC = 1
    PARA = 2
    SENT = 3

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
    if len(sent_lengths) <= 1:
        return None
    return variance(sent_lengths)

# given a list of words, return
# the variance of their length
def get_word_variance(words):
    word_lengths = []
    for word in words:
        word_lengths.append(len(word))
    return variance(word_lengths)

# returns a frequency dictionary of
# lemmas in the word list not including
# stop words
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

# return a frequency dictionary of
# bigrams in the wordlist
def get_bigrams(words):
    bgrams = bigrams([word.lower() for word in words])
    freqs = {}
    for gram in bgrams:
        if gram in freqs:
            freqs[gram] += 1
        else:
            freqs[gram] = 1
    return freqs

# download zip file
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

# gets repeated features for a topic classifier
def get_repeated_topic(sents, words, sw):
    features = {}
    features["sent_len"] = get_average_sentence_length(sents)
    features["word_len"] = get_average_word_length(sents)
    features["num_sents"] = len(sents)
    features["num_words"] = len(words)
    features["NNP"] = get_num_NNP(words)
    features["vehicle_dist"] = get_sim_count(words, "vehicle")
    features["person_dist"] = get_sim_count(words, "person")
    bow = get_bag_o_words(words, sw)
    bgrams = get_bigrams(words)
    features.update(bow)
    features.update(bgrams)
    return features

# gets repeated features for an author classifier
def get_repeated_author(sents, words, sw):
    features = {}
    features["sent_len"] = get_average_sentence_length(sents)
    features["word_len"] = get_average_word_length(sents)
    features["num_sents"] = len(sents)
    features["num_words"] = len(words)
    features["punct_dens"] = get_punc_density(sents)
    features["tobe"] = get_tobe_count(sents)
    features["sent_var"] = get_sent_variance(sents)
    features["word_var"] = get_word_variance(words)
    bow = get_bag_o_words(words, sw)
    bgrams = get_bigrams(words)
    features.update(bow)
    features.update(bgrams)
    return features

# get the features for a document, 
# topic classifier
def features_topic_doc(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            features = {}
            sents = sent_tokenize(raw)
            words = word_tokenize(raw)
            features = get_repeated_topic(sents, words, sw)
            if "A" in html:
                data.append((features, "A"))
            else:
                data.append((features, "B"))             
    return data

# get the features for a sentence, 
# topic classifier
def features_topic_sents(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            features = {}
            sents = sent_tokenize(raw)
            for sent in sents:
                words = word_tokenize(sent)
                features = get_repeated_topic(sent, words, sw)
                if "A" in html:
                    data.append((features, "A"))
                else:
                    data.append((features, "B"))             
    return data

# get the features for a paragraph, 
# topic classifier
def features_topic_paras(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            features = {}
            paras = parse_paragraphs(raw)
            for para in paras:
                sents = sent_tokenize(para)
                words = word_tokenize(para)
                features = get_repeated_topic(sents, words, sw)
                if "A" in html:
                    data.append((features, "A"))
                else:
                    data.append((features, "B"))             
    return data

# get the features for a
# document, author classifier
def features_author_doc(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            sents = sent_tokenize(raw)
            words = word_tokenize(raw)
            features = get_repeated_author(sents, words, sw)
            found = re.search("\d\d\d\d", html)
            data.append((features, found[0]))              
    return data

# get the features for a paragraph
# author classifier
def features_author_paras(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            paras = parse_paragraphs(raw)
            for para in paras:
                sents = sent_tokenize(para)
                words = word_tokenize(para)
                features = get_repeated_author(sents, words, sw)
                found = re.search("\d\d\d\d", html)
                data.append((features, found[0]))              
    return data

# get the features for the extra
# credit classifier
def features_author_paras_EC(directory):
    data = []
    sw = set(stopwords.words('english'))
    for html in os.listdir(directory):
        with open(os.path.join(directory, html), "r", encoding='utf-8') as file:
            raw = BeautifulSoup(file, 'html.parser').get_text()
            paras = parse_paragraphs(raw)
            for para in paras:
                sents = sent_tokenize(para)
                words = word_tokenize(para)
                features = get_repeated_author(sents, words, sw)
                found = re.search("\d\d\d\d", html)
                if "A" in html:
                    data.append((features, "A", found[0]))
                else:
                    data.append((features, "B", found[0]))
    return data

# get all training data from a directory
def get_features(directory, topic=True, ftype=FType.DOC):
    data = []
    if topic:
        if ftype == FType.DOC:
            data = features_topic_doc(directory)
        elif ftype == FType.PARA:
            data = features_topic_paras(directory)
        else:
            data = features_topic_sents(directory)
    else:
        if ftype == FType.DOC:
            data = features_author_doc(directory)
        elif ftype == FType.PARA:
            data = features_author_paras(directory)
    return data

# train and test a classifier on the given data
# print the associated metrics
def train_test_bayes(data):
    split = int(len(data) * TEST_SIZE)
    random.shuffle(data)
    train_set = data[split:]
    test_set = data[:split]
    test_features = [x[0] for x in test_set]
    test_categories = [x[1] for x in test_set]
    classifier = NaiveBayesClassifier.train(train_set)
    # classifier = DecisionTreeClassifier.train(train_set)
    # accuracy = get_accuracy(test_features, test_categories, classifier)
    # print(accuracy)
    print_metrics(test_features, test_categories, classifier)
    return classifier

# train and test the extra credit
# classifier.  This works slightly differently
# than the previous method. Also prints metrics
def train_test_extra_credit(data):
    train_set = [(x[0], x[2]) for x in data if x[1] == "A"]
    test_set = [(x[0], x[2]) for x in data if x[1] == "B"]
    test_features = [x[0] for x in test_set]
    test_categories = [x[1] for x in test_set]
    classifier = NaiveBayesClassifier.train(train_set)
    print_metrics(test_features, test_categories, classifier)
    return classifier

# prints accuracy, precision, recall and fscore
# for the given target set and response set
def print_metrics(test_features, test_categories, classifier):
    guesses = classifier.classify_many(test_features)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_categories, guesses, average='macro')
    print("------------------------------")
    print("Accuracy:\t\t" + str(get_accuracy(test_categories, guesses)))
    print("Precision:\t" + str(precision))
    print("Recall:\t\t\t" + str(recall))
    print("F-Score:\t\t" + str(fscore))
    print("------------------------------")

# get the accuracy of a response set
def get_accuracy(test_categories, guesses):
    correct = 0
    for i in range(len(guesses)):
        if test_categories[i] == guesses[i]:
            correct += 1
    return float(correct) / len(test_categories)

# given a text, separate into a list
# of paragraphs
def parse_paragraphs(text):
    paras = text.split("\n")
    filtered  = [x for x in paras if len(x) > 5]
    return filtered

# welcome
def main():
    import sys
    args = sys.argv[1:]
    if len(args) > 0:
        if (args[0] == "-DTC"):
            toggle = True
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    # download corpus
    download_zip(TRAINING_DATA, "training_set")
    download_zip(TRAINING_DATA_CURRENT, "training_set_current")

    # train and test each model
    print("------------------------------")
    print("Topic by Document")
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), topic=True, ftype=FType.DOC)
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")), topic=True, ftype=FType.DOC)
    data = data_1 + data_2
    tbd = train_test_bayes(data)
    
    print("Topic by Paragraph")
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), topic=True, ftype=FType.PARA)
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")), topic=True, ftype=FType.PARA)
    data = data_1 + data_2
    tbp = train_test_bayes(data)

    print("Topic by Sentence")
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), topic=True, ftype=FType.SENT)
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")), topic=True, ftype=FType.SENT)
    data = data_1 + data_2
    tbs = train_test_bayes(data)

    print("Author by Doc")
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), topic=False, ftype=FType.DOC)
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")), topic=False, ftype=FType.DOC)
    data = data_1 + data_2
    abd = train_test_bayes(data)

    print("Author by Paragraph")
    data_1 = get_features(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")), topic=False, ftype=FType.PARA)
    data_2 = get_features(os.path.join(".", os.path.join("training_set", "proj1F21_files")), topic=False, ftype=FType.PARA)
    data = data_1 + data_2
    abp = train_test_bayes(data)

    print("Author by Paragraph Extra Credit")
    data_1 = features_author_paras_EC(os.path.join(".", os.path.join("training_set_current", "proj1S23_files")))
    data_2 = features_author_paras_EC(os.path.join(".", os.path.join("training_set", "proj1F21_files")))
    data = data_1 + data_2
    ec = train_test_extra_credit(data)

    # load data for examples
    data_tbd = get_features("./examples/docs", topic=True, ftype=FType.DOC)
    data_tbp = get_features("./examples/paras", topic=True, ftype=FType.PARA)
    data_tbs = get_features("./examples/sents", topic=True, ftype=FType.SENT)
    data_abd = get_features("./examples/docs", topic=False, ftype=FType.DOC)
    data_abp = get_features("./examples/paras", topic=False, ftype=FType.PARA)
    data_ec = get_features("./examples/paras", topic=False, ftype=FType.PARA)

    # run examples through classifier
    print("----------------------")
    print("Car Document Topic Example")
    print(tbd.classify(data_tbd[0][0]))
    print("----------------------")
    print("Car Paragraph Topic Example")
    print(tbp.classify(data_tbp[0][0]))
    print("----------------------")
    print("Person Sentence Topic Example")
    print(tbs.classify(data_tbs[0][0]))
    print("----------------------")
    print("Car Document Author Example")
    print(abd.classify(data_abd[0][0]))
    print("----------------------")
    print("Car Paragraph Author Example")
    print(abp.classify(data_abp[0][0]))
    print("----------------------")
    print("Car Paragraph Author Example EC")
    print(ec.classify(data_ec[0][0]))
    print("----------------------")

if __name__=="__main__":
    main()