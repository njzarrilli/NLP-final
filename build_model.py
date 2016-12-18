import re
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
import numpy as np
import string

rubric = { '1': 12, '3': 3, '4': 3, '5': 4, '6':4, '7': 30, '8':60 }

def train_model(training_data, training_results):
    pass


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
        for word in essay.split():
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
        for word in data[i].split():
            try:
                index = vocabulary_index_dict[word.lower()]
                table[i][index] += 1
            except KeyError:
                pass
    
    return table

def log_accuracy(predictions, grades):
    grade_accuracies = defaultdict(lambda: [0.0, 0.0])
    total = 0
    correct_predictions = 0
    f = open("testing_accuracy_results.txt", "w+")
    
    gradeCounts = defaultdict(int)
    for predicted_grade, correct_grade in zip(predictions, grades):
        f.write("Got: %s    Expected: %s\n" % (predicted_grade, correct_grade))
        total += 1
        grade_accuracies[correct_grade][1] += 1
        if predicted_grade == correct_grade:
            correct_predictions += 1
            grade_accuracies[correct_grade][0] +=1
        gradeCounts[correct_grade] += 1
    
    accuracy = (float(correct_predictions)/total)*100
    f.write("Accuracy: %s \n\n" % str(accuracy))
    for grade in grade_accuracies:
        f.write("For %s correctly predicted %s essay grades out of %s\n" % (grade, grade_accuracies[grade][0], grade_accuracies[grade][1]))
    f.write("\n")
    f.close()
    print gradeCounts


def main():
    f = open("training_set_rel3.tsv")
    lines = list(f)
    train_data_essays, train_data_scores, train_score_dict = get_training_data(lines)
    vocabulary = get_vocabulary(train_data_essays)
    table = bag_of_words(train_data_essays, len(vocabulary), vocabulary)
    print train_score_dict
    f.close()

    f = open("test_set.tsv")
    lines = list(f)
    test_data_essays = get_test_data(lines)
    f.close()

    tfidf_transformer = TfidfTransformer(use_idf=False).fit(table)
    X_train_tf = tfidf_transformer.transform(table)
    logreg = linear_model.LogisticRegression()
    logreg.fit(X_train_tf, train_data_scores)

    best_model = linear_model.LogisticRegression(tol=.000001)

    docs_new = test_data_essays
    
    # testing on training data to check for accuracy
    table_test = bag_of_words(test_data_essays, len(vocabulary), vocabulary)
    X_new_tfidf = tfidf_transformer.transform(table_test)
    predictions = logreg.predict(X_new_tfidf)
    log_accuracy(predictions, train_data_scores)

main()