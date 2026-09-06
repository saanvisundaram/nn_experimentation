import numpy as np
import struct
import csv
import string
from sklearn.feature_extraction.text import TfidfVectorizer as tfv
from sklearn.model_selection import train_test_split as tts
import torch
from torch.utils.data import Dataset, DataLoader


filename = 'GoogleNews-vectors-negative300.bin'
translator = str.maketrans('', '', string.punctuation)


def parse_word2vec_binary(bin_file_path, words, limit=None):
    word_vectors = {}
    
    # Open the file in binary read mode ('rb')
    with open(bin_file_path, 'rb') as f:
        # 1. Read the text header line
        header = f.readline().decode('utf-8').strip()
        vocab_size, vector_dim = map(int, header.split())
        print(f"Vocabulary Size: {vocab_size}")
        print(f"Vector Dimensions: {vector_dim}")
        
        # Calculate how many bytes each float vector consumes (4 bytes per float)
        vector_bytes_size = vector_dim * 4
        
        # Cap the loop if you only want a subset (e.g., first 10,000 words)
        words_to_read = min(vocab_size, limit) if limit else vocab_size
        
        # 2. Iterate and unpack each word entry
        for i in range(words_to_read):
            word_bytes = b""
            
            # Read character by character until finding the space delimiter
            while True:
                ch = f.read(1)
                if ch == b' ':
                    break
                if ch == b'': # End of file fail-safe
                    break
                word_bytes += ch
            
            word = word_bytes.decode('utf-8', errors='ignore').strip()

            # include possible break if word dne in review database
            if (word not in words):
                f.seek(vector_bytes_size, 1)
                continue
            
            # Read the raw binary float coefficients
            raw_vector = f.read(vector_bytes_size)
            
            # Unpack the binary string into a tuple of standard Python floats ('f' stands for float32)
            vector_tuple = struct.unpack(f"{vector_dim}f", raw_vector)
            
            # Store in your dictionary
            word_vectors[word] = list(vector_tuple)
            
            # Optional progress counter for huge datasets like GoogleNews
            if i > 0 and i % 10000 == 0:
                print(f"Successfully processed {i} words...")

    return word_vectors

def clean_up_text(sentence):
    # Remove punctuation
    cleaned_text = sentence.translate(translator)
    return cleaned_text # Output: Hello World Hows it going

def word2vec(sentence) -> list:
    fsv = [0.0] * 300
    sentence = clean_up_text(sentence.lower())
    for w in sentence.split(" "):
        if w in word_embeddings:
            embed = word_embeddings[w]
            fsv = np.add(fsv, embed)
    
    # fsv = fsv / len(sentence.split(" "))
    fsv = [val / len(sentence.split(" ")) for val in fsv]

    return fsv

# Load & randomise dataset here
data_train = []
labels_train = []
num_sentences = 5500
num_testing = 500


# read all dataset stuff
with open('IMDB Dataset.csv', mode='r', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    header = next(reader) 

    all_words = set()
    
    for _ in range(num_sentences):
        try:
            row = next(reader)
            clean_sent = clean_up_text(row[0].lower())
            data_train.append(clean_sent)
            labels_train.append(row[1])
            all_words.update(clean_sent.split(" "))
        except StopIteration:
            # Code hits this if the CSV has fewer than 10500 rows total
            break

print("here 1")


word_embeddings = parse_word2vec_binary('GoogleNews-vectors-negative300.bin', all_words)

# Example Usage:
# print("Vector for 'apple':", type(word_embeddings.get('apple')))

# first_sent = "hello how are you"
# fsv = [0.0] * 300
# for w in first_sent.split(" "):
#     if w in word_embeddings:
#         embed = word_embeddings[w]
#         fsv = np.add(fsv, embed)

# print(fsv)


print('here 2')

for label in range(num_sentences):
    if labels_train[label] == 'positive':
        labels_train[label] = 1
    else: labels_train[label] = 0

vectorizer = tfv(lowercase=True, max_features = 1000)

data_train_tfidf = vectorizer.fit_transform(data_train).toarray()
# data_train_fit = vectorizer.fit(data_train)
# data_train_transform = vectorizer.transform(data_train)

data_train_word2vec = []
for sentence in data_train:
    data_train_word2vec.append(word2vec(sentence))

print('here 3')

labels_train = np.reshape(labels_train, (num_sentences, 1))
all_data = np.hstack((data_train_word2vec, labels_train))

tfidf_word2vec_data = np.hstack((data_train_tfidf, all_data))

np.random.seed(42) # For reproducibility
np.random.shuffle(tfidf_word2vec_data)

np.savetxt("word2vec_train.txt", tfidf_word2vec_data[0:(num_sentences - num_testing)], delimiter=",")
np.savetxt("word2vec_test.txt", tfidf_word2vec_data[(num_sentences - num_testing):], delimiter=",")

print('Done!')


############################################ TFIDF RAW CODE ############################################
