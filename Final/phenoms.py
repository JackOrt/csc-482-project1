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
    import pprint
    temp_list = []
    i = 0
    for i in range(7):
        i+=1
        temp_list.append(db.getDiscussion(i))

    common_sense_detector(temp_list)

def common_sense_detector(hearing):
    for discussion in hearing:
        for text in discussion:
            print(text)



if __name__ == "__main__":
    main()