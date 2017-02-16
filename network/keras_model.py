import numpy as np
# from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras import backend
import pickle
import sys

import seq2seq
from seq2seq.models import SimpleSeq2Seq


np.random.seed(1337)  # for reproducibility

max_features = 26
maxlen = 8  # cut texts after this number of words
batch_size = 10

training_data = pickle.load(open("dirty/toy_data/toy_embeddings.pkl", "rb"))
questions = training_data[:50]
answers = training_data[50:]

val_samples = np.random.randint(0,50,10)
X_val, y_val = [], []
for sample in val_samples:
  X_val.append(questions[sample][:,:])
  y_val.append(answers[sample][:,:])

(X_train, y_train) = np.asarray(questions), np.asarray(answers)
X_val, y_val = np.asarray(X_val), np.asarray(y_val)

# print('Pad sequences (samples x time)')
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_val = sequence.pad_sequences(X_val, maxlen=maxlen)
# print('X_train shape:', X_train.shape)
# print('X_val shape:', X_val.shape)
# print('y_val shape:', y_val.shape)

print('Building model...')
# model = Sequential()
model = SimpleSeq2Seq(input_dim=26, hidden_dim=50, output_length=8, output_dim=26)
model.compile(loss='mse', optimizer='adam')
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'perplexity'])
lookup = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

for i in xrange(10):
  print('Training...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=128,
      validation_data=(X_val, y_val), verbose=0)
  score = model.evaluate(X_val, y_val, batch_size=batch_size)
  # print('Test score:', score)

  test_samples = np.random.choice(50, 10, replace=False)

  for sample in test_samples:
    test = X_train[sample,:,:]
    preds = model.predict(np.asarray([test]), verbose=0)[0]
    predictions = preds.tolist()

    result = ['Prediction: ']
    for letter in test:
      try:
        position = letter.tolist().index(1)
        result.append(lookup[position])
      except ValueError:
        result.append(' ')
    result.append(' ')
    for letter in predictions:
      big = max(letter)
      position = letter.index(big)
      # print letter
      if big > 0.5:
        result.append(lookup[position])
      elif big > 0.4:
        result.append("("+lookup[position]+")")
      else:
        result.append('-')

    print ''.join(result)




# for i in range(8):
#   x = np.zeros((1, maxlen, len(chars)))
#   for t, char in enumerate(sentence):
#       x[0, t, char_indices[char]] = 1.


#   next_index = sample(preds, diversity=0.5)
#   next_char = indices_char[next_index]

#   generated += next_char
#   sentence = sentence[1:] + next_char
