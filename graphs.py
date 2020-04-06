import matplotlib.pyplot as plt
import csv


def build_plot(dataset, vectorizer, classifier):
    dim = []
    acc = []
    pre = []
    rec = []
    f1 = []
    for row in dataset:
        if row['vectorizer'] == vectorizer and row['classifier'] == classifier:
            dim.append(float(row['dimensions']))
            acc.append(float(row['accuracy']))
            pre.append(float(row['precision']))
            rec.append(float(row['recall']))
            f1.append(float(row['f1']))
    return {
        'dimensions': dim,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1
    }


filename = 'tabulated.csv'
with open(filename) as csv_file:
    dataset = csv.DictReader(csv_file, delimiter=',')
    #w2v_cnn = build_plot(dataset, 'GloVe', 'CNN')
    #csv_file.seek(0)
    #w2v_lstm = build_plot(dataset, 'GloVe', 'LSTM')
    #csv_file.seek(0)
    w2v_double = build_plot(dataset, 'W2V', 'CNN+LSTM')
    csv_file.seek(0)
    glove_double = build_plot(dataset, 'GloVe', 'CNN+LSTM')


#plt.plot(w2v_cnn['dimensions'], w2v_cnn['recall'], 'k-', label='CNN')
#plt.plot(w2v_lstm['dimensions'], w2v_lstm['recall'], 'k--', label='LSTM')
plt.plot(w2v_double['dimensions'], w2v_double['accuracy'], 'k-', label='Word2Vec')
plt.plot(glove_double['dimensions'], glove_double['accuracy'], 'k--', label='Glove')
plt.xlabel('Dimensões')
plt.ylabel('Acurácia')
plt.title('Acurácia CNN + LSTM')
plt.legend()
plt.show()
