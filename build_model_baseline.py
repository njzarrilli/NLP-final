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
import random
#jar = '/Users/njzarrilli/GitHub/NLP-final/stanford-postagger-2015-12-09/stanford-postagger-3.6.0.jar'
#model = 'Users/njzarrilli/GitHub/NLP-final/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger'
#tagger = StanfordPOSTagger(model, jar)


rubric = { '1': 12, '3': 3, '4': 3, '5': 4, '6':4, '7': 30, '8':60 }
entities = {"organization": '<ORG>', 'location' : '<LOC>', 'caps':'<CAPS>', 
            'num': '<NUM>', 'percent': '<PERCENT>', 'person': '<PERS>', 
            'month': '<MONTH>', 'date': '<DATE>', 'money': '<MONEY>' }

UNKNOWN_TOKEN = "<UNK>"

def get_test_data(lines):
    test_data_lines = []

    for line in lines[1:]:
        line_data = re.split(r'\t+', line)
        line_data[2] = line_data[2].decode('unicode_escape').encode('ascii','ignore')
        essay = line_data[2].encode('utf-8')
        essay = essay.translate(None, string.punctuation)
        essay = essay.lower()
        test_data_lines.append(essay)
    
    return (test_data_lines)

def get_vocabulary(data):
    vocabulary = set()

    for essay in data:
        for word in essay:
            vocabulary.add(word)
            # ignore punctuation
    return vocabulary

# add in unknown token as a word
def bag_of_words(data, vocab_size, vocabulary):
    table = np.zeros((len(data), vocab_size))
    vocabulary_list = list(vocabulary)
    vocabulary_index_dict = dict()
    for k in range(len(vocabulary_list)):
        word = vocabulary_list[k]
        vocabulary_index_dict[word] = k

    for i in range(len(data)):
        # do this in preprocessing
        for word in data[i]:
            try:
                index = vocabulary_index_dict[word.lower()]
                table[i][index] += 1
            except KeyError:
                pass
    return table

def get_bigram_vocab(data):
    bigram_vocab = set()

    for essay in data:
        for i in range(1, len(essay)):
            bigram = (essay[i-1], essay[i])
            bigram_vocab.add(bigram)

    return bigram_vocab


def log_accuracy(predictions, grades):
    grade_accuracies = defaultdict(lambda: [0.0, 0.0])
    total = 0
    correct_predictions = 0
    f = open("testing_accuracy_baseline.txt", "w+")
    
    gradeCounts = defaultdict(int)
    for predicted_grade, correct_grade in zip(predictions, grades):
        f.write("Got: %s    Expected: %s\n" % (predicted_grade, correct_grade))
        total += 1
        grade_accuracies[correct_grade][1] += 1
        if predicted_grade == correct_grade:
            correct_predictions += 1
            grade_accuracies[correct_grade][0] += 1
        gradeCounts[correct_grade] += 1
    
    accuracy = (float(correct_predictions)/total)*100
    print("Accuracy: %s \n\n" % str(accuracy))
    f.write("Accuracy: %s \n\n" % str(accuracy))
    for grade in grade_accuracies:
        f.write("For %s correctly predicted %s essay grades out of %s\n" % (grade, grade_accuracies[grade][0], grade_accuracies[grade][1]))
    f.write("\n")
    f.close()
    print gradeCounts

def main():

    f = open("essays_randomized.txt")
    lines = list(f)
    print("hi")
    essays_tokenize = [word_tokenize(essay) for essay in lines]
    f.close()
    print("hi")
    
    f = open("scores_randomized.txt")
    scores = list(f)
    f.close()

    # pos_file = open("essays_tagged_randomized.txt")
    # pos_lines = list(pos_file)
    # pos_file.close()
    print len(essays_tokenize), len(scores)
    
    train_data_essays, train_data_scores = essays_tokenize[:10178], scores[:10178]
    test_data_essays, test_data_scores = essays_tokenize[10178:], scores[10178:]
    
    
    # training
    vocabulary = get_vocabulary(train_data_essays)
    print("vocab found")
    bigram_vocabulary = get_bigram_vocab(train_data_essays)
    BOW_matrix = bag_of_words(train_data_essays, len(vocabulary), vocabulary)
    clf = MultinomialNB().fit(BOW_matrix, train_data_scores)
    print("about to fit logreg model")
    print("finished training")
    
    # testing on training data to check for accuracy
    BOW_matrix = bag_of_words(test_data_essays, len(vocabulary), vocabulary)
    print("about to predict")
    predictions = clf.predict(BOW_matrix)
    log_accuracy(predictions, test_data_scores)

main()