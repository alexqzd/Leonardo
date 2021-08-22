import numpy as np
import json
import pickle

# # Loading json data
# print("Loading json data...")
# with open('intents.json') as file:
#   data = json.loads(file.read())

# data = np.array(data['data'])
# data = data.T

# text = data[0]
# labels = data[1]

# text = list((map(lambda x: x.lower(), text)))

# Loading csv data
import csv, random

text = list()
labels = list()
rows = list()

print("Loading csv data...")
with open('intents.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        rows.append(row)
random.shuffle(rows)
for row in rows:
    text.append(row[0])
    labels.append(row[1])

text = list((map(lambda x: x.lower(), text)))


from sklearn.model_selection import train_test_split
train_txt,test_txt,train_label,test_labels = train_test_split(text,labels,test_size = 0.3)


# Since deep learning is a game of numbers, it’d expect our data to be in numerical form to play with.
# We shall tokenize our dataset;
# meaning break the sentences into individuals and convert these individuals into numerical representations.

print("Tokenizing data...")

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
max_num_words = 40000
classes = np.unique(labels)

print(classes)

tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(train_txt)
word_index = tokenizer.word_index

# To feed our data to the deep learning model, all our phrases must be of same length.
# We shall pad all our training phrases with 0 so that they become of same length.

print("Padding data...")

ls=[]
for c in train_txt:
    ls.append(len(c.split()))
maxLen=int(np.percentile(ls, 98))
train_sequences = tokenizer.texts_to_sequences(train_txt)
train_sequences = pad_sequences(train_sequences, maxlen=maxLen, padding='post')
test_sequences = tokenizer.texts_to_sequences(test_txt)
test_sequences = pad_sequences(test_sequences, maxlen=maxLen, padding='post')

# We shall now convert our labels into one-hot vectors.

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(classes)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)
train_label_encoded = label_encoder.transform(train_label)
train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
train_label = onehot_encoder.transform(train_label_encoded)
test_labels_encoded = label_encoder.transform(test_labels)
test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
test_labels = onehot_encoder.transform(test_labels_encoded)

# Before we begin training our model, we shall use the Global Vectors.
# Since it is trained on a large corpus, it will help the model to learn the phrases even better.

print("Loading GloVe data...")

embeddings_index={}

with open('utils/GloVe.pkl','rb') as file:
  embeddings_index = pickle.load(file)

# with open('SBW-vectors-300-min5.txt', encoding='utf8') as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs

# with open('utils/GloVe.pkl','wb') as file:
#    pickle.dump(embeddings_index,file)


# Since GloVe contains vector representation of all the words from a large corpus,
# we’ll need only those word vectors that are present in our corpus.
# We shall create an embedding matrix that contains the vector representations
# of only the words that are present in our dataset.

print("Creating embedding matrix of the words present in our dataset...")

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
num_words = min(max_num_words, len(word_index))+1
embedding_dim=len(embeddings_index['el'])
embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_num_words:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# We shall now create our model.

print("Creating model...")

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional,Embedding
model = Sequential()

model.add(Embedding(num_words, 300, trainable=False,input_length=train_sequences.shape[1], weights=[embedding_matrix]))
model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(classes.shape[0], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# We shall now train our model.

print("Training model...")

history = model.fit(train_sequences, train_label, epochs = 30,
          batch_size = 8, shuffle=True,
          validation_data=[test_sequences, test_labels])

# Visualize the metrics

print("Visualizing metrics...")

import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

# Saving Model, Tokenizer, Label Encoder and Labels

print("Saving model...")


model.save('models/intents.h5')

with open('utils/classes.pkl','wb') as file:
   pickle.dump(classes,file)

with open('utils/tokenizer.pkl','wb') as file:
   pickle.dump(tokenizer,file)

with open('utils/label_encoder.pkl','wb') as file:
   pickle.dump(label_encoder,file)

print("Done!")