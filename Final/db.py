from dotenv import load_dotenv
import os
import pymysql.cursors
import time
#First, we install our DDDB Python interface, PyMySQL
#https://pypi.org/project/PyMySQL/


# DATABASE credentials
#This is a read-only user/password to the Amazon RDB instance: 
#DO NOT SHARE!!! DO NOT MAKE THIS NOTEBOOK PUBLIC!!!
connection = None
creds = {}

def init():
    load_dotenv()
    myhost = os.getenv("HOST")
    myuser = os.getenv("USER")
    mypass = os.getenv("PASS")
    mydb = os.getenv("DB")
    connection = pymysql.connect(
    host=myhost, user=myuser, password=mypass, database=mydb
    )
    creds["myhost"] = myhost
    creds["myuser"] = myuser
    creds["mypass"] = mypass
    creds["mydb"] = mydb

########################## prints total number of utterances ##############################
#Testing
# sql = "SELECT count(*) FROM Utterance"

# with connection:
#     with connection.cursor() as cursor:
#       cursor.execute(sql)
#       result = cursor.fetchall()

# for res in result:
#   print("Total number of utterances in database: ",str(res[0]))

###############This is code that gives all utterances of all legislators (extremely expensive, limited to 100) #####################
# connection = pymysql.connect(
#   host=creds["myhost"], user=creds["myuser"], password=mypass, database=mydb
# )

# sql2 = """
# SELECT U.uid, U.pid, P.first, P.last, U.did, U.text 
# FROM Utterance as U 
# LEFT JOIN Person as P on U.pid = P.pid
# LEFT JOIN PersonClassifications C on U.pid = C.pid
# WHERE U.state = 'CA' AND current = 1 AND finalized = 1 AND C.PersonType = "Legislator"
# LIMIT 10
# """

# with connection:
#     with connection.cursor() as cursor:
#       cursor.execute(sql2)
#       result = cursor.fetchall()
#     for res in result:
#       print(str(res))
    

####################This function returns a list of utterances, which is everything that was said by the person in DDDB CA ########################
#Input: LegislatorID who is a valid ID of a California Legislator
def getLegUtterances(LegislatorID):
  connection = pymysql.connect(
    host=creds["myhost"], user=creds["myuser"], password=creds["mypass"], database=creds["mydb"]
  )
  sqltest = """
SELECT U.uid, U.pid, P.first, P.last, U.did, U.text 
FROM Utterance as U 
LEFT JOIN Person as P on U.pid = P.pid
LEFT JOIN PersonClassifications as C on U.pid = C.pid
WHERE U.pid = 1 AND U.state = 'CA' AND current = 1 AND finalized = 1 AND C.PersonType = 'Legislator'
"""

  sql2 = """
SELECT U.text
FROM Utterance as U
LEFT JOIN Person as P on U.pid = P.pid
WHERE U.pid = %s AND 
    U.state = 'CA' AND 
    current = 1 AND 
    finalized = 1 AND 
    U.pid in (SELECT pid FROM Legislator)
  """

  with connection:
    with connection.cursor() as cursor:
      cursor.execute(sql2,(str(LegislatorID),))
      result = cursor.fetchall()
  return [r[0] for r in result]

##### test for previous function
# leg10 = getLegUtterances(10)
# print("Testing... legislator pid=10 has",len(leg10),"utterances listed below\n",leg10)

#This Function returns how every legislator voted in a bill discussion, 
# it returns a list of (pid, vote) tuples given a single bill discussion ID as input

def getVotes(DiscussionID):
  connection = pymysql.connect(
    host=creds["myhost"], user=creds["myuser"], password=creds["mypass"], database=creds["mydb"]
  )

  sql = """ SELECT v.pid, v.result
      FROM BillDiscussion bd, Hearing h, BillVoteDetail v, BillVoteSummary s 
      WHERE bd.did = %s AND
            bd.hid = h.hid AND
            h.date = s.voteDate AND
            s.cid in (SELECT cid from CommitteeHearings c where c.hid = h.hid) AND
            bd.bid = s.bid AND
            v.voteId = s.voteId AND
            v.state = 'CA'
        """
  with connection:
    with connection.cursor() as cursor:
      cursor.execute(sql,(str(DiscussionID),))
      result = cursor.fetchall()
  return [r for r in result]

#Test for bill discussion #136
# getVotes(136)


def getDiscussion(DiscussionID):
  connection = pymysql.connect(
    host=creds["myhost"], user=creds["myuser"], password=creds["mypass"], database=creds["mydb"]
  )

  sqltest = """
				select h.hid, bd.did, uid, v.vid, v.fileId as vid_file_id, p.pid, p.first, p.last, time, u.endTime, 
				u.type, text, alignment, h.state, h.date
                from Utterance u, Hearing h, Video v, BillDiscussion bd, Person p
                where bd.did = % s and
                      h.hid = bd.hid and
                      bd.did = u.did and
                      v.vid = u.vid and
                      u.pid = p.pid and
                      u.current = 1 and finalized = 1
                order by vid, time
LIMIT 100
"""
  with connection:
    with connection.cursor() as cursor:
      cursor.execute(sqltest,str(DiscussionID))
      result = cursor.fetchall()
      print(result)

# test for discussion_id 7
# getDiscussion(7)

