from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data_dir = Path('dataset', 'Labelled')
to_split = 'dataset\\Merged_classified\\revised.csv'

df = pd.read_csv(to_split)
df_tweet_label = df[['class_est', 'id', 'tweet']]

train, test = train_test_split(
    df_tweet_label,
    test_size=0.25,
    random_state=42,
    shuffle=True
)

train_path = Path(data_dir, '2class_training_br.csv')
test_path = Path(data_dir, '2class_testing_br.csv')

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)