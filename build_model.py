import re
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict


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
        test_data_lines.append(essay)
    
    return (test_data_lines)

# def bag_of_words(data):
#     # wordToId = dict()
#     # idToOccur = dict()
#     table = defaultdict(int)
    
#     for essay in data:
#         for word in essay:
#             table[(essay, word)] += 1
#     return table

def main():
    f = open("training_set_rel3.tsv")
    lines = list(f)
    train_data_essays, train_data_scores, train_score_dict = get_training_data(lines)
    # print train_score_dict
    
    # return 
    # print(train_data_essays[0])
    f.close()

    f = open("test_set.tsv")
    lines = list(f)
    test_data_essays = get_test_data(lines)
    # print test_data_essays[0]
    f.close()



    count_vect = CountVectorizer(min_df=1, ngram_range=(1,3))
    X_train_counts = count_vect.fit_transform(train_data_essays)
    print(X_train_counts.shape)

    tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tfidf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)

    best_model = linear_model.LogisticRegression(tol=.000001)

    docs_new = test_data_essays
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_train_counts)
    #traing model on train data and test it on the held out data set
    best_model = train_model(train_data_essays, train_data_scores)
    # print count_vect.vocabulary_

    clf = MultinomialNB().fit(X_train_tf, train_data_scores)

    gradeCounts = defaultdict(int)
    predicted = clf.predict(X_new_tfidf)
    for doc, grade in zip(docs_new, predicted):
        gradeCounts[grade] += 1
        # print doc, grade
    print gradeCounts


main()