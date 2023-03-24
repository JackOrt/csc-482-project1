import db
from nltk import word_tokenize
from nltk import download 
from nltk import Text
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag

I_TEXT = 11
I_FILEID = 4
I_TIME = 8
NEG_SENT_CUTOFF = -0.9
POS_SENT_CUTOFF = 0.9

###### potential phenoms #########
# http://ai4reporters.org/CA_201720180AB447_96_54363_53047
# bill has no opposition (no no votes)
# aye <name>: aye or nay
# measure passes
# reader stutters, seems to have trouble reading
# Some summaries are much longer than others
# "shouldn't even be having this discussion"
# extreme rhetoric (example from diabetes)
# "common sense bill"
# SOME bills gain much more support than others
# calling absent voters multiple times

# senator yields time to another to introduce bill
# move to table the bill
# laughter

def main():
    target = "climate"

    db.init()
    hearing = db.getHearing(54363)
    print(entity_detector(hearing))
    occ = negative_sentiment_detector(hearing)
    for i in range(50):
        hearing = db.getHearing(i)
        occ_positive = positive_sentiment_detector(hearing)
        found, occ_word_utterances = get_word_utterances(hearing, target)
        
        if len(occ_positive) > 0:
            print("Printing URLs for positive sentiments in hearing: " + str(i))

        for oc in occ_positive:
            if oc != None:
                print(formatVid(oc[I_FILEID], oc[I_TIME]))        

        if len(occ_word_utterances) > 0:
            if found:
                print("Printing URLs for utterances of " + target + " in hearing: " + str(i))

        for oc in occ_word_utterances:
            if oc != None:
                print(formatVid(oc[I_FILEID], oc[I_TIME]))

    # hid = db.getHearingID("H4IEb-n5ABk")
    # print(hid)
    # hearings = []
    # i = 0
    # for i in range(99):
    #     i+=1
    #     hearings.append(db.getHearing(i))

    # test = db.getHearing(72)
    # for hearing in hearings:
    #     occurences = common_sense_detector(hearing)
    #     if len(occurences) > 0:
    #         print(occurences)

# given a hearing, returns the utterances with potential statistics
def stats_detector(hearing):
    banned = ["SB", "item", "Item", "Bill", "SCR", "Items", "items"]
    block = False
    occurences = []
    for utterance in hearing:
        sentences = sent_tokenize(utterance[I_TEXT])
        for sentence in sentences:
            pos_type = pos_tag(word_tokenize(sentence))
            for word in banned:
                if word in sentence:
                    block = True
                    break
            if not block: 
                for i in range(len(pos_type)):
                    if i != 0:
                        prev, tag = pos_type[i - 1]
                        word, tag = pos_type[i]
                        if tag == "CD" and prev not in banned:
                            occurences.append(sentence)
    return occurences

def get_utterance_text(utterances, text):
    for utterance in utterances:
        if text in utterance[I_TEXT]:
            return utterance

def get_word_utterances(hearing, query):
    utterances = []
    combined_utterances = []
    found = False
    for utterance in hearing:
        combined_utterances += word_tokenize(utterance[I_TEXT])
    
    utterance_text = Text(combined_utterances)
    contexts = utterance_text.concordance_list(query)
    if contexts is not None:
        for context in contexts:
            utterances.append(get_utterance_text(hearing, context.line.strip()))
            found = True
            if utterances[0] == None:
                found = False

    return found, utterances

def common_sense_detector(hearing):
    occurences = defaultdict(list)
    for utterance in hearing:
        if "common sense" in utterance[I_TEXT]:
            occurences.append(utterance)
    return occurences

# given a hearing, return the utterances with strong negative sentiment
def negative_sentiment_detector(hearing):
    occurences = []
    sia = SentimentIntensityAnalyzer()
    for utt in hearing:
        sents = sent_tokenize(utt[I_TEXT])
        comp = 0
        # find average negative sentament over an utterance
        for sent in sents:
            scores = sia.polarity_scores(sent)
            comp += scores['compound']
        comp /= len(sents)
        if comp < NEG_SENT_CUTOFF:
            occurences.append(utt)
    return occurences

def positive_sentiment_detector(hearing):
    occurences = []
    sia = SentimentIntensityAnalyzer()
    for utt in hearing:
        sents = sent_tokenize(utt[I_TEXT])
        comp = 0
        # find average negative sentament over an utterance
        for sent in sents:
            scores = sia.polarity_scores(sent)
            comp += scores['compound']
        comp /= len(sents)
        if comp > POS_SENT_CUTOFF:
            occurences.append(utt)
    return occurences
        
# return a set of named entities in a hearing
def entity_detector(hearing):
    banned = ["File", "Pass", "Aye", "Was", "Good", "Amen", "Part", "Please", "Has", "INAUDIBLE"]
    entities = set()
    sia = SentimentIntensityAnalyzer()
    for utt in hearing:
        sents = sent_tokenize(utt[I_TEXT])
        for sent in sents:
            words = word_tokenize(sent)
            for word in banned:
                if word in words:
                    words.remove(word)
            tagged = pos_tag(words)
            ne_tree = ne_chunk(tagged)
            if len(words) > 3:
                ne_list = [c.leaves() for c in ne_tree.subtrees(lambda s: s.label() != "PERSON" and len(s.label()) > 1)]
                for entity in ne_list:
                    new_entity = " ".join([e[0] for e in entity])
                    entities.add(new_entity)
    return entities

#formats video as a link 
def formatVid(id, offset=0):
    if len(id) < 12:
        vlink = "https://youtu.be/"+id+"?"
    else:
        vlink = f"https://videostorage-us-west.s3-us-west-2.amazonaws.com/videos/{id}/{id}.mp4#"
    
    if offset > 0:
        vlink += "t="+str(offset)

    return vlink

def formatTime(offset):
    hours, minutes, seconds = 0,0,0
    hours = "00"
    if offset > 3600:
        hours = offset // 3600
        if hours > 9:
            hours = str(hours)
        else:
            hours = "0"+str(hours)

    offset = offset % 3600
    minutes = offset // 60
    if minutes < 10:
        minutes = "0"+str(minutes)
    else:
        minutes = str(minutes)

    offset = offset % 60
    seconds = offset
    if seconds < 10:
        seconds = "0"+str(seconds)
    else:
        seconds = str(seconds)

    return str(hours)+":"+minutes+":"+seconds


if __name__ == "__main__":
    download('vader_lexicon')
    download('punkt')
    download('maxent_ne_chunker')
    download('words')
    main()