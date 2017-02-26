import numpy as np
import tensorflow as tf
import os
import time
import pickle
import sys

from utils.general import Progbar, init_generator
from utils.parser import minibatches
from utils.getEmbeddings import get_batches, loader
from tensorflow.contrib.layers import xavier_initializer

class Config(object):
  n_cells = 40      # number cells units in RNN layer passed into rnn.GRUCell()
  max_enc_len = 7       #theoretically, not needed with dynamic RNN
  max_dec_len = 8           # purposely different from enc to easily distinguish
  vocab_size = 27      # 26 letters of the alphabet and 1 for padding
  embed_size = 27
  dropout_rate = 0.9
  n_epochs = 3
  learning_rate = 0.001
  initializer = "glorot" # for xavier or "normal" for truncated normal
  batch_size = 10

class Seq2SeqModel(object):
  def add_placeholders(self):
    # (batch_size, sequence_length, embedding_dimension)
    self.input_placeholder = tf.placeholder(tf.float32,
        shape=(self.batch_size, 7, self.embed_size), name='question')
    # (batch_size, sequence_length, vocab_size)
    self.output_placeholder = tf.placeholder(tf.float32,
        shape=(self.batch_size, 8, self.embed_size), name='answer')
    self.enc_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.dec_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.labels = tf.placeholder(tf.int32, shape=(self.batch_size, self.max_dec_len))
    self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

  def create_feed_dict(self, input_data, output_data, labels, sequence_length):
    # When output_data is None, that means we are in inference mode making
    # making predictions, rather than fitting the data.  However, the model
    # will not run unless there is target output_data, so we generate bunch
    # of fake data for the output and the labels
    if output_data is None:
      output_data = np.zeros((self.batch_size, 8, self.vocab_size))
      labels = np.zeros((self.batch_size,self.max_dec_len))

    feed_dict = {
      self.input_placeholder: input_data,
      self.output_placeholder: output_data,
      self.enc_seq_len: sequence_length["enc"],
      self.dec_seq_len: sequence_length["dec"],
      self.labels: labels,
      self.dropout_placeholder: self.dropout_rate
    }

    return feed_dict

  # Sequence_length is a vector (or list) that has length equal to the batch_size
  # For example, suppose you have a 50 examples, split into 5 batches, then
  # your vector should have a length of 10. Additionally, each sentence in your
  # batch varies in the number of words, up to 15 words.  Finally, each word
  # embedding requires 40 dimensions.  Then, an example sequence_length might be:
  #     [8, 7, 8, 6, 13, 12, 11, 10, 15, 7]
  # This means the first sentence has 8 words in it, second sentence has words in
  # it, the third sentence has 8 words as well.  Notice that neither 40 nor 50 show
  # up anywhere, that was a trick question.

  def encoder_decoder(self):
    init_state = tf.get_variable('init_state', [self.batch_size, self.n_cells],
         initializer=tf.contrib.layers.xavier_initializer())

    with tf.variable_scope("seq2seq") as scope:
      enc_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      dec_cell = tf.contrib.rnn.GRUCell(self.n_cells)

      # Encoder
      _, enc_state = tf.nn.dynamic_rnn(enc_cell,
          self.input_placeholder, sequence_length=self.enc_seq_len,
          initial_state=init_state, dtype=tf.float32)
      # Intermediate decoder function
      decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
      # Decoder
      stage = "inference" if self.labels is None else "fitting"
      if stage is "fitting":
        with tf.variable_scope("decoder"):
          pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
              dec_cell, decoder_fn=decoder_fn,
              inputs=self.output_placeholder, sequence_length=self.dec_seq_len)
        with tf.variable_scope("decoder", reuse=True):
          pred, dec_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
              dec_cell, decoder_fn=decoder_fn,
              inputs=self.output_placeholder, sequence_length=self.dec_seq_len)
      elif self.stage is "inference":
        with tf.variable_scope("decoder"):
          pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
              decoder_fn=decoder_fn, inputs=None, sequence_length=self.dec_seq_len)
        with tf.variable_scope("decoder", reuse=True):
          pred, dec_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
              dec_cell, decoder_fn=decoder_fn, inputs=None,
              sequence_length=self.dec_seq_len)

    return pred, dec_state

  def add_loss_op(self, pred):
    """Adds Ops for the loss function to the computational graph.
    Args:
        pred: A tensor of shape (batch_size, n_classes)
    Returns:
        loss: A 0-d tensor (scalar) output

    """
    # loss = tf.nn.sequence_loss(logits, self.output_placeholder, weights)
    # (logits, targets, weights, average_across_timesteps=True,
    #   average_across_batch=True, softmax_loss_function=None, name=None):

    # idx = 10 * 8 + (7-1) = 80 + 6 = 86
    # idx = tf.range(self.batch_size)*tf.shape(pred)[1] + (self.max_dec_len - 1)
    # tf.reshape flattens the multi-dimensional tensor, then tf.gather grabs the
    # final output in the list, which we call the "last_output"
    # last_output = tf.gather(tf.reshape(pred, [-1, self.n_cells]), idx)
    # this is another way to do the same thing
    # last_output = tf.gather_nd(logits,
    #     tf.pack([tf.range(50), self.max_dec_len-1], axis=1))
    # we end up not wanting just the last element since we are making a series
    # of predictions, rather than just a single predictions
    # pred = tf.Print(pred, [tf.shape(pred)], first_n=3, message="before pad:") #[10, 8, 40]

    # for some reason, when a particular batch has sequence length less
    # than the max, the prediction are truncated to the max of that batch
    # rather than maintainingthe same length, so we
    # add in those as zeros appended to the end of the tensor
    diff = self.max_dec_len - tf.shape(pred)[1]
    # paddings is [   [dim1 before, dim1 after],
    #                 [dim2 before, dim2 after],
    #                 [dim3 before, dim3 after]   ]
    paddings = [[0,0], [0,diff], [0,0]]
    pred = tf.pad(pred, paddings, mode='CONSTANT', name="pad")

    # pred = tf.Print(pred, [tf.shape(pred)], first_n=3, message="after pad:") #[10, 8, 40]

    flat_size = self.max_dec_len * self.n_cells
    flattened_preds = tf.reshape(pred, [self.batch_size, flat_size])
    # now the predictions are shape (10, 8*40)

    # logits of shape [batch_size, seq_len, vocab_size]
    # labels of shape [batch_size, seq_len].

    with tf.variable_scope('lossy', initializer=xavier_initializer()):
      flat_vocab_size = self.max_dec_len * self.vocab_size
      weight = tf.get_variable(name="W", shape=(flat_size, flat_vocab_size))
      bias = tf.get_variable(name="b", shape=(flat_vocab_size,) )
      # flat_logits should have shape (batch_size, flat_vocab_size) (10, 208)
      flat_logits = tf.matmul(flattened_preds, weight) + bias
      # logits should have shape (10, 8, 26)
      logits = tf.reshape(flat_logits,
          [self.batch_size, self.max_dec_len, self.vocab_size])
      final_output = tf.nn.softmax(logits)

    # final_output = tf.Print(final_output, [tf.shape(final_output)],
    #     first_n=3, message="Final output shape")

    # tf.summary.histogram("WeightLossy",weight)
    # tf.summary.histogram("biasLossy",bias)
    # tf.summary.histogram("logitsLossy", logits)

    # we don't pass in the final_output here since the loss function
    # already includes the softmax calculation inside
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.labels, logits=logits)
    tf.summary.histogram("cross_entropy_loss", cross_entropy_loss)
    loss = tf.reduce_mean(cross_entropy_loss)
    return loss, final_output

  def add_training_op(self, loss):
    optimizer = tf.train.AdamOptimizer(self.lr)
    train_op = optimizer.minimize(loss)
    tf.summary.scalar("loss", loss)
    return train_op

  def predict(self, sess, test_samples):
    lookup = list('abcdefghijklmnopqrstuvwxyz')
    seq_len= {"enc": [np.sum(sen) for sen in test_samples],
        "dec": [8 for sen in test_samples]}
    _, final_output = sess.run([self.loss, self.final_output],
        self.create_feed_dict(test_samples, None, None, seq_len) )
    embedding_to_text(test_samples, final_output)

  def train(self, sess, summary_op):
    allBatches = get_batches(self.all_data, self.batch_size, False, True)
    prog = Progbar(target=(len(self.all_data)/2) / self.batch_size)
    fetches = [self.train_op, self.loss, summary_op]    # array of desired outputs

    for i, batch in enumerate(allBatches):
      questions, answers = batch[0], batch[1]
      enc_seq_len = [np.sum(sen) for sen in questions]
      dec_seq_len = [np.sum(sen) for sen in answers]
      seq_len = {"enc": enc_seq_len, "dec": dec_seq_len}
      labels = []
      for word in answers:
        label = [letter.index(1) for letter in word]
        labels.append(label)
      labels = np.asarray(labels)

      feed_dict = self.create_feed_dict(questions, answers, labels, seq_len)
      _, loss, summary = sess.run(fetches, feed_dict)
      prog.update(i + 1, [("train loss", loss)])
    # return summary

  def build(self):
    self.add_placeholders()
    self.pred, self.dec_state = self.encoder_decoder()
    self.loss, self.final_output = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)

  def __init__(self, config, training_data):
    self.n_cells = config.n_cells
    self.max_enc_len = config.max_enc_len
    self.max_dec_len = config.max_dec_len
    self.embed_size = config.embed_size
    self.vocab_size = config.vocab_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.initializer = config.initializer
    self.dropout_rate = config.dropout_rate
    self.all_data = training_data
    self.batch_size = config.batch_size
    self.build()

def embedding_to_text(test_samples, final_output):
  lookup = list(' abcdefghijklmnopqrstuvwxyz')
  for idx, sample_word in enumerate(test_samples):
    result = ['Prediction ', str(idx+1), ': ']

    for letter in sample_word:
      try:
        position = letter.index(1)
        result.append(lookup[position])
      except ValueError:
        result.append(' ')
    result.append(' ')

    predicted_word = final_output[idx]
    for letter in predicted_word:
      big = max(letter)
      position = letter.tolist().index(big)
      if big > 0.5:
        result.append(lookup[position])
      elif big > 0.4:
        result.append("("+lookup[position]+")")
      else:
        result.append('-')
    print ''.join(result)

def print_bar(stage):
  print 80 * "="
  print stage.upper()
  print 80 * "="

def main(debug=True):
  config = Config()
  all_data = pickle.load(open("dirty/toy_data/toy_embeddings.pkl", "rb"))

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    model = Seq2SeqModel(config, all_data)
    print "took {:.2f} seconds\n".format(time.time() - start)
    # saver = None if debug else tf.train.Saver()

    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      summary_op = tf.summary.merge_all()

      print_bar("training")
      for epoch in range(model.n_epochs):
        logs_path = '/tmp/tensorflow/board'       # Tensorboard. Run: tensorboard --logdir=run1:/tmp/tensorflow/board --port 6006
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)
        model.train(session, summary_op)
        # writer.add_summary(summary, epoch)# * batch_count + i)

      print_bar("prediction")
      test_indices = np.random.choice(50, 10, replace=False)
      test_samples = [all_data[i] for i in test_indices]
      predictions = model.predict(session, test_samples)

if __name__ == '__main__':
    main()
