import matplotlib.pyplot as plt
import csv


def build_plot(dataset, vectorizer, classifier):
    tt = []
    dim = []
    acc = []
    pre = []
    rec = []
    f1 = []
    for row in dataset:
        if row['vectorizer'] == vectorizer and row['classifier'] == classifier:
            dim.append(float(row['dimensions']))
            tt.append(float(row['train_time (s)']))
            acc.append(float(row['accuracy'])*100)
            pre.append(float(row['precision'])*100)
            rec.append(float(row['recall'])*100)
            f1.append(float(row['f1'])*100)
    return {
        'dimensions': dim,
        'train': tt,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }


filename = 'data_paper.csv'
with open(filename) as csv_file:
    dataset = csv.DictReader(csv_file, delimiter=',')
    # w2v_cnn = build_plot(dataset, 'W2V', 'CNN')
    # csv_file.seek(0)
    # w2v_lstm = build_plot(dataset, 'W2V', 'LSTM')
    # csv_file.seek(0)
    w2v_double = build_plot(dataset, 'W2V', 'CNN+LSTM')
    # csv_file.seek(0)
    # glove_cnn = build_plot(dataset, 'GloVe', 'CNN')
    # csv_file.seek(0)
    # glove_lstm = build_plot(dataset, 'GloVe', 'LSTM')
    csv_file.seek(0)
    glove_double = build_plot(dataset, 'GloVe', 'CNN+LSTM')
    #csv_file.seek(0)
    #fast_double = build_plot(dataset, 'fastText', 'CNN')

# plt.plot(w2v_cnn['dimensions'], w2v_cnn['accuracy'], 'r-', label='CNN')
# plt.plot(w2v_lstm['dimensions'], w2v_lstm['accuracy'], 'g--', label='LSTM')
# plt.plot(w2v_double['dimensions'], w2v_double['accuracy'], 'b:', label='CNN + LSTM')
# plt.plot(w2v_cnn['dimensions'], w2v_cnn['precision'], 'r-', label='CNN')
# plt.plot(w2v_lstm['dimensions'], w2v_lstm['precision'], 'g--', label='LSTM')
# plt.plot(w2v_double['dimensions'], w2v_double['precision'], 'b:', label='CNN + LSTM')
# plt.plot(w2v_cnn['dimensions'], w2v_cnn['recall'], 'r-', label='CNN')
# plt.plot(w2v_lstm['dimensions'], w2v_lstm['recall'], 'g--', label='LSTM')
# plt.plot(w2v_double['dimensions'], w2v_double['recall'], 'b:', label='CNN + LSTM')
# plt.plot(glove_cnn['dimensions'], glove_cnn['accuracy'], 'r-', label='CNN')
# plt.plot(glove_lstm['dimensions'], glove_lstm['accuracy'], 'g--', label='LSTM')
# plt.plot(glove_double['dimensions'], glove_double['accuracy'], 'b:', label='CNN + LSTM')
# plt.plot(glove_cnn['dimensions'], glove_cnn['precision'], 'r-', label='CNN')
# plt.plot(glove_lstm['dimensions'], glove_lstm['precision'], 'g--', label='LSTM')
# plt.plot(glove_double['dimensions'], glove_double['precision'], 'b:', label='CNN + LSTM')
# plt.plot(glove_cnn['dimensions'], glove_cnn['recall'], 'r-', label='CNN')
# plt.plot(glove_lstm['dimensions'], glove_lstm['recall'], 'g--', label='LSTM')
# plt.plot(glove_double['dimensions'], glove_double['recall'], 'b:', label='CNN + LSTM')
# plt.plot(w2v_double['dimensions'], w2v_double['accuracy'], 'g-', label='Word2Vec')
# plt.plot(glove_double['dimensions'], glove_double['accuracy'], 'b--', label='GloVe')
# plt.plot(fast_cnn['dimensions'], fast_cnn['accuracy'], 'r:', label='fastText')
# plt.plot(w2v_double['dimensions'], w2v_double['precision'], 'g-', label='Word2Vec')
# plt.plot(glove_double['dimensions'], glove_double['precision'], 'b--', label='GloVe')
# plt.plot(fast_lstm['dimensions'], fast_lstm['precision'], 'r:', label='fastText')
plt.plot(w2v_double['dimensions'], w2v_double['recall'], 'g-', label='Word2Vec')
plt.plot(glove_double['dimensions'], glove_double['recall'], 'b--', label='GloVe')
# plt.plot(fast_double['dimensions'], fast_double['recall'], 'r:', label='fastText')
plt.xlabel('Dimensões')
plt.ylabel('Recall')
plt.ylim(94, 96.5)
plt.legend()
plt.show()
