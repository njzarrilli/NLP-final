import re
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import string
import nltk
from nltk.tag.stanford import StanfordPOSTagger
from nltk.internals import find_jars_within_path
from nltk import word_tokenize
import os

# os.environ['CLASSPATH']='~/stanford-parser/stanford-parser.jar'
# os.environ['STANFORD_MODELS']='~/stanford-ner/classifiers'
entities = {"organization": '<ORG>', 'location' : '<LOC>', 'caps':'<CAPS>', 
            'num': '<NUM>', 'percent': '<PERCENT>', 'person': '<PERS>', 
            'month': '<MONTH>', 'date': '<DATE>', 'money': '<MONEY>' }

UNKNOWN_TOKEN = "<UNK>"

def NER(essay):
    to_remove = []

    for i in range(len(essay)):
        try:
            if essay[i] == "@":
                entity = re.sub("\d+", "", essay[i+1])
                # print essay[i], essay[i+1], entity
                essay[i] = entities[entity]
                to_remove.append(essay[i+1])
        except KeyError:
            pass
    for i in to_remove:
        essay.remove(i)
    return essay

# raw = "Dear local newspaper, I think effects computers have on people are great learning skills/affects because they give us time to chat with friends/new people, helps us learn about the globe(astronomy) and keeps us out of troble! Thing about! Dont you think so? How would you feel if your teenager is always on the phone with friends! Do you ever time to chat with your friends or buisness partner about things. Well now - there's a new way to chat the computer, theirs plenty of sites on the internet to do so: @ORGANIZATION1, @ORGANIZATION2, @CAPS1, facebook, myspace ect. Just think now while your setting up meeting with your boss on the computer, your teenager is having fun on the phone not rushing to get off cause you want to use it. How did you learn about other countrys/states outside of yours? Well I have by computer/internet, it's a new way to learn about what going on in our time! You might think your child spends a lot of time on the computer, but ask them so question about the economy, sea floor spreading or even about the @DATE1's you'll be surprise at how much he/she knows. Believe it or not the computer is much interesting then in class all day reading out of books. If your child is home on your computer or at a local library, it's better than being out with friends being fresh, or being perpressured to doing something they know isnt right. You might not know where your child is, @CAPS2 forbidde in a hospital bed because of a drive-by. Rather than your child on the computer learning, chatting or just playing games, safe and sound in your home or community place. Now I hope you have reached a point to understand and agree with me, because computers can have great effects on you or child because it gives us time to chat with friends/new people, helps us learn about the globe and believe or not keeps us out of troble. Thank you for listening."
raw = "Please parse this, thanks."
essay = raw.lower()

essay = word_tokenize(essay)
essay = NER(essay)
essay = " ".join(essay)
essays = [essay]

POS_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=postagger --jar=$HOME/stanford-parser/stanford-parser.jar --modeljar=$HOME/stanford-parser/stanford-parser-3.5.2-models.jar --model=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz --input=essay.txt"
lex_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=lexparser --jar=$HOME/stanford-postagger/stanford-postagger.jar --model=$HOME/stanford-postagger/models/english-bidirectional-distsim.tagger --input=essay.txt"
ner_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=nertagger --jar=$HOME/stanford-ner/stanford-ner.jar --model=$HOME/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz --input=essay.txt"
for essay in essays:
        file = open("essay-parse-dec-12.txt", 'w')
        file.write(essay)
        file.close()
        # bash += ' ' + "--model=$HOME/stanford-postagger/models/english-bidirectional-distsim.tagger"
        # print bash
        print os.system(lex_bash)
