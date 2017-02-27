import numpy as np
import pandas as pd
import pickle
import sys
import re

def pad(word, total):
    wl = len(word)
    needed = total - wl
    padding = "-" * needed
    return word + padding

def clean_up(raw_half, size):
    half = []
    for idx, word in enumerate(raw_half):
      # print word
      padded = pad(word, size)
      ascii_word = map(ord, padded)
      numpyWord = np.zeros( (len(ascii_word), 27) )
      for idx, letter in enumerate(ascii_word):
        if letter > 50:
          position = letter - 97
          position += 1 # since pad is a position
          numpyWord[idx][position] = 1
        else:
          numpyWord[idx][0] = 1
      half.append(numpyWord.tolist())
      # print half
    return half

if __name__ == '__main__':
  raw_data = []
  with open ("phrases.txt", "r") as myfile:
      raw_data = myfile.read().splitlines()

  data = [phrase.split() for phrase in raw_data]
  data = zip(*data)

  raw_enc = list(data[0])
  raw_dec = list(data[1])

  enc = clean_up(raw_enc, 7)
  dec = clean_up(raw_dec, 8)

  final_data = enc + dec
  with open('toy_embeddings_new.pkl', 'wb') as filename:
    pickle.dump(final_data, filename)
