import csv

filename = 'C:\\Users\\yami_\\workspace\\tweet_collector\\classified.csv'
with open(filename, 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    resultado = {
        'relevantes': 0,
        'não relevantes': 0
    }
    for tweet in reader:
        if tweet[0] == '1':
            resultado['relevantes'] += 1
        else:
            resultado['não relevantes'] += 1

print(resultado)
