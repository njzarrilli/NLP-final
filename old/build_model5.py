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
import ast

#jar = '/Users/njzarrilli/GitHub/NLP-final/stanford-postagger-2015-12-09/stanford-postagger-3.6.0.jar'
#model = 'Users/njzarrilli/GitHub/NLP-final/stanford-postagger-2015-12-09/models/english-bidirectional-distsim.tagger'
#tagger = StanfordPOSTagger(model, jar)


rubric = { '1': 12, '3': 3, '4': 3, '5': 4, '6':4, '7': 30, '8':60 }
entities = {"organization": '<ORG>', 'location' : '<LOC>', 'caps':'<CAPS>', 
            'num': '<NUM>', 'percent': '<PERCENT>', 'person': '<PERS>', 
            'month': '<MONTH>', 'date': '<DATE>', 'money': '<MONEY>' }

UNKNOWN_TOKEN = "<UNK>"
# ORGANIZATION
# LOCATION
# CAPS
# NUM
# PERCENT
# PERSON
# MONTH
# DATE
# MONEY

def detect_unknowns(data, word_counts):
    data_with_unknowns = []

    for essay in data:
        new_essay = []
        for word in essay:
            if word_counts[word] <= 1:
                new_essay.append(UNKNOWN_TOKEN)
            else:
                new_essay.append(word)
        data_with_unknowns.append(new_essay)
    
    return data_with_unknowns

def NER(essay):
    to_remove = []
    for i in range(len(essay) - 1):
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
            

def letter_grade(essay_set, essay_score):
    essay_score_int = int(essay_score)

    if essay_set == '1':
        if essay_score == '5' or essay_score == '6':
            return 'A'
        elif essay_score == '4':
            return 'B'
        elif essay_score == '3':
            return 'C'
        elif essay_score == '2':
            return 'D'
        else:
            return 'F'
    # might not be out of 4 might be 3
    elif essay_set == '3' or essay_set == '4':
        if essay_score == '3':
            return 'A'
        elif essay_score == '2':
            return 'B'
        elif essay_score == '1':
            return 'D'
        else:
            return 'F'
    elif essay_set == '5' or essay_set == '6':
        if essay_score == '4':
            return 'A'
        elif essay_score == '3':
            return 'B'
        elif essay_score == '2':
            return 'C'
        elif essay_score == '1':
            return 'D'
        else:
            return 'F'
    elif essay_set == '7':
        if essay_score_int >= 24:
            return 'A'
        elif essay_score_int >= 18:
            return 'B'
        elif essay_score_int >= 12:
            return 'D'
        else:
            return 'F'
    elif essay_set == '8':
        if essay_score_int >= 50:
            return 'A'
        elif essay_score_int >= 40:
            return 'B'
        elif essay_score_int >= 30:
            return 'C'
        elif essay_score_int >= 20:
            return 'D'
        else:
            return 'F'

def randomize(essays, scores, POSfile):
    print 'start randomize'
    # print "Len essays", len(essays)
    # print "Len scores", len(scores)
    # print "Len pos", len(POSfile)

    essays_unt = []
    scores_unt = []
    essays_tagged = []
    combined = list(zip(essays, POSfile, scores))
    i = 0 
  
    random.shuffle(combined)
    # for essay in combined:
        # i += 1
        # tagged = ast.literal_eval(essay[1])
        # try: 
            # if essay[0][0] != tagged[0][0]:
                # print essay[0][0], tagged[0][0]
                # print "===============\n\n"
                # print "fails at:", i
            # if (i % 100) == 0:
                # print i
        # except:
            # print essay
            # print tagged
    essays_unt[:], essays_tagged[:], scores_unt[:] = zip(*combined)
    return (essays_unt, essays_tagged, scores_unt)

def get_training_data(lines):
    train_data_lines = []
    train_result_lines = []
    gradeCounts = defaultdict(int)
    word_counts = defaultdict(int)
    lines = lines[1:]

    for line in lines:

        line_data = re.split(r'\t+', line)
        line_data[2] = line_data[2].decode('unicode_escape').encode('ascii','ignore')
        
        essay = line_data[2].encode('utf-8')
        #essay = essay.translate(None, string.punctuation)
        essay = essay.lower()
        essay = word_tokenize(essay)
        essay = essay[1:-1] # remove quotes
        essay = NER(essay)  # replace NE with tokens
        essay_score = line_data[5]
        #essay = essay.split()

        essay_set = line_data[1].encode('utf-8')
        
        if essay_set != '2':
            normalized_score = int(float(essay_score) / rubric[essay_set] * 100)
            grade = letter_grade(essay_set, essay_score)
            for word in essay:
                word_counts[word] += 1

            train_data_lines.append(essay)
            train_result_lines.append(grade)
            
            gradeCounts[grade] += 1

    train_data_lines = detect_unknowns(train_data_lines, word_counts)

    return (train_data_lines, train_result_lines, gradeCounts)



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

def bag_of_pos(tagged, num_tags, tags):
    table = np.zeros((len(tagged), num_tags))
    tag_list = list(tags)
    tag_index_dict = dict()

    for k in range(len(tag_list)):
        tag = tag_list[k]
        tag_index_dict[tag] = k

    for i in range(len(data)):
        for word in data[i]:
            try:
                index = tag_index_dict[word[1]]
                table[o][index] += 1
            except KeyError:
                pass
    return table

def bag_of_NE(data):
    table = np.zeros((len(data), len(entities)))
    entities_index_dict = dict()
    k=0
    for entity in entities:
        entities_index_dict[k] = entity
        k += 1

    for i in range(len(data)):
        for word in data[i]:
            try:
                index = entities_index_dict[word.lower()]
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

def get_tag_list(tagged):
    tag_list = set()

    for essay in tagged:
        essay = ast.literal_eval(essay)
        for word in essay:
            print word
            tag_list.add(word[1])

    return tag_list

def bag_of_bigrams(data, bigram_vocab_size, bigram_vocabulary):
    table = np.zeros((len(data), bigram_vocab_size))
    bigram_list = list(bigram_vocabulary)
    bigram_index_dict = dict()
    
    for k in range(len(bigram_list)):
        bigram = bigram_list[k]
        bigram_index_dict[bigram] = k

    for i in range(len(data)):
        essay = data[i]
        for j in range(1,len(essay)):
            try:
                bigram = (essay[j-1].lower(), essay[j].lower())
                index = bigram_index_dict[bigram]
                table[i][index] += 1
            except KeyError:
                pass
    
    return table

def log_accuracy(predictions, grades):
    grade_accuracies = defaultdict(lambda: [0.0, 0.0])
    total = 0
    correct_predictions = 0
    f = open("testing_accuracy_grades_pos.txt", "w+")
    
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

def cosine_similarities(data_matrix, compare_matrix):
    cosine_sim_matrix = cosine_similarity(data_matrix, compare_matrix)
    print(cosine_sim_matrix.shape)
    final_matrix = np.concatenate((data_matrix.toarray(), cosine_sim_matrix), axis=1)
    print(final_matrix.shape)
    return final_matrix


def POS_tag_feature(text):
    return tagger.tag(text.split())

def essay_sizes(data):
    essay_lengths = []

    for essay in data:
        essay_lengths.append(len(essay))

    essay_lengths = np.reshape(essay_lengths, (len(data),1))
    return essay_lengths

def features_matrix(data, data_tagged, vocabulary, bigram_vocabulary, tag_list, tfidf_transformer=None):
    
    #BOW_matrix = bag_of_words(data, len(vocabulary), vocabulary)
    #print("bow found")
    BOB_matrix = bag_of_bigrams(data, len(bigram_vocabulary), bigram_vocabulary)
    print("bob found")
    #bags_matrix = np.concatenate((BOW_matrix, BOB_matrix), axis=1)  
    #print("concat")
    NE_matrix = bag_of_NE(data)
    print("ne found")
    combined_matrix1 = np.concatenate((BOB_matrix, NE_matrix), axis=1)
    print('concat 1')

    # POS_matrix = bag_of_pos(data_tagged, len(tag_list), tag_list)
    # print("POS found")
    # combined_matrix2 = np.concatenate((combined_matrix1, POS_matrix), axis=1)
    # print('concat 2')
    combined_matrix2 = combined_matrix1
    if not tfidf_transformer:
        tfidf_transformer = TfidfTransformer().fit(combined_matrix2)
        print("tfidf found")
    X_train_tf = tfidf_transformer.transform(combined_matrix2)
    # essay_lengths = essay_sizes(data)
    # final_matrix = np.concatenate((X_train_tf, essay_lengths), axis=1)
    return (tfidf_transformer, X_train_tf)

def main():

    f = open("training_set_rel3.tsv")
    lines = list(f)
    
    # pos_file = open("training_set_tagged.txt")
    pos_file = open("all_data_tagged.txt")
    pos_lines = list(pos_file)

    essays, scores, train_score_dict = get_training_data(lines)
    essays_randomized, scores_randomized, essays_tagged_randomized = randomize(essays, scores, pos_lines)
    
    train_data_essays, train_data_scores, train_data_tagged = essays_randomized[:10178], scores_randomized[:10178], essays_tagged_randomized[:10178]
    test_data_essays, test_data_scores, test_data_tagged = essays_randomized[10178:], scores_randomized[10178:], essays_tagged_randomized[10178:]
    

    print train_data_essays[0]
    print train_score_dict
    
    
    # training
    vocabulary = get_vocabulary(train_data_essays)
    print("vocab found")
    bigram_vocabulary = get_bigram_vocab(train_data_essays)
    # tag_list = get_tag_list(train_data_tagged)
    tag_list = []
    print("got tag list")
    
    tfidf_transformer, train_matrix = features_matrix(train_data_essays, train_data_tagged, vocabulary, bigram_vocabulary, tag_list)
    logreg = linear_model.LogisticRegression(max_iter=1000)
    print("about to fit logreg model")
    logreg.fit(train_matrix, train_data_scores)
    print("finished training")
    
    # testing on training data to check for accuracy
    tfidf_transformer, test_matrix = features_matrix(test_data_essays, test_data_tagged, vocabulary, bigram_vocabulary, tag_list, tfidf_transformer)
    print("about to predict")
    predictions = logreg.predict(test_matrix)
    log_accuracy(predictions, test_data_scores)

main()
