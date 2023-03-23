import db
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
    hid = db.getHearingID("H4IEb-n5ABk")
    print(hid)
    import pprint
    hearings = []
    i = 0
    for i in range(99):
        i+=1
        hearings.append(db.getHearing(i))

    test = db.getHearing(72)
    for hearing in hearings:
        occurences = common_sense_detector(hearing)
        if len(occurences) > 0:
            print(occurences)

def common_sense_detector(hearing):
    occurences = []
    for utterance in hearing:
        if "common sense" in utterance[11]:
            occurences.append(utterance)
    return occurences

if __name__ == "__main__":
    main()