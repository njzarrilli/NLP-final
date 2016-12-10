import re
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def train_model(training_data, training_results):
    pass


def get_training_data(lines):
    train_data_lines = []
    train_result_lines = []

    for line in lines[1:]:
        line_data = re.split(r'\t+', line)
        line_data[2] = line_data[2].decode('unicode_escape').encode('ascii','ignore')
        essay = line_data[2].encode('utf-8')
        essay_score = line_data[5]
        train_data_lines.append(essay)
        train_result_lines.append(essay_score)
    
    return (train_data_lines, train_result_lines)

def main():
    f = open("training_set_rel3.tsv")
    lines = list(f)
    train_data_essays, train_data_scores = get_training_data(lines)
    print(train_data_essays[0])
    #print(train_data_scores[:2])
   

    count_vect = CountVectorizer(min_df=1)
    X_train_counts = count_vect.fit_transform(train_data_essays)
    print("hello")
    print(X_train_counts.shape)

    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)

    best_model = linear_model.LogisticRegression(tol=.000001)


    #traing model on train data and test it on the held out data set
    best_model = train_model(train_data_essays, train_data_scores)

main()