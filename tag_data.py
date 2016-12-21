
import re
import os
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import string
from nltk.tag.stanford import StanfordPOSTagger
from nltk.internals import find_jars_within_path

rubric = { '1': 12, '3': 3, '4': 3, '5': 4, '6':4, '7': 30, '8':60 }
jar = '/Users/AlexanderRavan/GitHub/NLP-final/stanford-postagger-2015-12-09/stanford-postagger-3.6.0.jar'
model = '/Users/AlexanderRavan/GitHub/NLP-final/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger'
tagger = StanfordPOSTagger(model, jar)


def get_training_data(lines):
	train_data_lines = []
	train_result_lines = []
	gradeCounts = defaultdict(int)

	for line in lines[1:]:

		line_data = re.split(r'\t+', line)
		line_data[2] = line_data[2].decode('unicode_escape').encode('ascii','ignore')
		
		essay = line_data[2].encode('utf-8')
		essay = essay.translate(None, string.punctuation)
		essay = essay.lower()
		essay_score = line_data[5]
		
		essay_set = line_data[1].encode('utf-8')
		
		if essay_set != '2':
			normalized_score = int(float(essay_score) / rubric[essay_set] * 100)
			if normalized_score > 80:
				grade = 'A'
			elif normalized_score <= 80 and normalized_score > 65:
				grade = 'B'
			elif normalized_score <= 65 and normalized_score > 50:
				grade = 'C'
			elif normalized_score <= 50 and normalized_score > 45:
				grade = 'D'
			else:
				grade = 'F'

			train_data_lines.append(essay)
			train_result_lines.append(grade)
			
			gradeCounts[grade] += 1

	return (train_data_lines, train_result_lines, gradeCounts)

def main():
	f = open("training_set_rel3.tsv")
	lines = list(f)
	essays, scores, train_score_dict = get_training_data(lines)
	
	train_data_essays, train_data_scores = essays[:10178], scores[:10178]


	# POS_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=postagger --jar=$HOME/stanford-parser/stanford-parser.jar --modeljar=$HOME/stanford-parser/stanford-parser-3.5.2-models.jar --model=edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz --input=essay.txt"
	# lex_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=lexparser --jar=$HOME/stanford-postagger/stanford-postagger.jar --model=$HOME/stanford-postagger/models/english-bidirectional-distsim.tagger --input=essay.txt"
	# ner_bash = "python ~/GitHub/NLP-final/nltk_cli/stanford.py --tool=nertagger --jar=$HOME/stanford-ner/stanford-ner.jar --model=$HOME/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz --input=essay.txt"

	tagged_essays = open('tagged_essays.txt', "w")
	for essay in train_data_essays[550:]:
		# file = open("essay.txt", 'w')
		# file.write(essay)
		# file.close()
		# bash += ' ' + "--model=$HOME/stanford-postagger/models/english-bidirectional-distsim.tagger"
		# print bash
		# tagged = os.system(POS_bash)
		tagged_essays.write(str(tagger.tag(essay.split())))
		tagged_essays.write('\n')
	tagged_essays.close()

main()



