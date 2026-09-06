import csv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
from sklearn.model_selection import train_test_split as tts
import torch
from torch.utils.data import Dataset, DataLoader

print("here 1")
# Load & randomise dataset here
data_train = []
labels_train = []


with open('IMDB Dataset.csv', mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    header = next(reader) 
    
    for _ in range(5500):
        try:
            row = next(reader)
            data_train.append(row[0])
            labels_train.append(row[1])
        except StopIteration:
            # Code hits this if the CSV has fewer than 50 rows total
            break

for label in range(5500):
    if labels_train[label] == 'positive':
        labels_train[label] = 1
    else: labels_train[label] = 0





print("Here 2")
# making dataset with tfidf
vectorizer = tfv(lowercase=True, max_features = 1000)

data_train_tfidf = vectorizer.fit_transform(data_train).toarray()
# data_train_fit = vectorizer.fit(data_train)
# data_train_transform = vectorizer.transform(data_train)

print("here 3")
labels_train = np.reshape(labels_train, (-1, 1))
all_data = np.hstack((data_train_tfidf, labels_train))

print(all_data.shape)
print(all_data)

test_sets = 30
np.savetxt("tfidf_reviews_train.txt", all_data[0:5000], delimiter=",")
np.savetxt("tfidf_reviews_test.txt", all_data[5000:], delimiter=",")