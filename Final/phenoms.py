import db
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

I_TEXT = 11
I_FILEID = 4
I_TIME = 8
NEG_SENT_CUTOFF = -0.95
POS_SENT_CUTOFF = 0.95

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
    db.init()
    hearing = db.getHearing(54363)
    occ = negative_sentiment_detector(hearing)
    for i in range(500):
        occ = positive_sentiment_detector(db.getHearing(i))
        for oc in occ:
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

def common_sense_detector(hearing):
    occurences = []
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
    main()