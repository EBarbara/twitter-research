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
            acc.append(float(row['accuracy']))
            pre.append(float(row['precision']))
            rec.append(float(row['recall']))
            f1.append(float(row['f1']))
    return {
        'dimensions': dim,
        'train': tt,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }


filename = 'tabulated.csv'
with open(filename) as csv_file:
    dataset = csv.DictReader(csv_file, delimiter=',')
    #w2v_cnn = build_plot(dataset, 'W2V', 'CNN')
    #csv_file.seek(0)
    #w2v_lstm = build_plot(dataset, 'W2V', 'LSTM')
    #csv_file.seek(0)
    w2v_double = build_plot(dataset, 'W2V', 'CNN')
    csv_file.seek(0)
    glove_double = build_plot(dataset, 'GloVe', 'CNN')
    csv_file.seek(0)
    fast_double = build_plot(dataset, 'fastText', 'CNN')


#plt.plot(w2v_cnn['dimensions'], w2v_cnn['recall'], 'k-', label='CNN')
#plt.plot(w2v_lstm['dimensions'], w2v_lstm['recall'], 'k--', label='LSTM')
plt.plot(w2v_double['dimensions'], w2v_double['train'], 'k-', label='Word2Vec')
#plt.plot(glove_cnn['dimensions'], glove_cnn['accuracy'], 'k-', label='CNN')
#plt.plot(glove_lstm['dimensions'], glove_lstm['accuracy'], 'k--', label='LSTM')
plt.plot(glove_double['dimensions'], glove_double['train'], 'k--', label='GloVe')
#plt.plot(fast_cnn['dimensions'], fast_cnn['accuracy'], 'k-', label='CNN')
#plt.plot(fast_lstm['dimensions'], fast_lstm['accuracy'], 'k--', label='LSTM')
plt.plot(fast_double['dimensions'], fast_double['train'], 'k-*', label='fastText')
plt.xlabel('Dimensions')
plt.ylabel('Train time')
plt.legend()
plt.show()
