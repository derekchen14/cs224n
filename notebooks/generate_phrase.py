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
      numpyWord = np.zeros( (len(ascii_word), 29) )
      for idx, letter in enumerate(ascii_word):
        if letter > 70:
          position = letter - 97
          position += 1         # since pad is a position
          numpyWord[idx][position] = 1
        elif letter == 45:      # "-" token used for padding
          numpyWord[idx][0] = 1
        elif letter == 60:     # "<" token used for Start-of-Word
          numpyWord[idx][27] = 1
        elif letter == 62:     # ">" token used for End-of-Word
          numpyWord[idx][28] = 1
      half.append(numpyWord.tolist())
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
  dec = clean_up(raw_dec, 10)

  final_data = enc + dec
  with open('toy_embeddings.pkl', 'wb') as filename:
    pickle.dump(final_data, filename)
