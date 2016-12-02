def train_model(training_data, training_results):
    pass

def get_held_out_data(lines):
    pass

def get_training_data(lines):
    pass

def main():
    f = open("training_set_rel3.tsv")
    lines = list(f)
    train_data_lines, train_result_lines = get_training_data(lines)
    held_out_data_lines, held_out_result_lines = get_held_out_data(lines)

    #traing model on train data and test it on the held out data set
    best_model = train_model(train_data_lines, train_result_lines)