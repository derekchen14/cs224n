import numpy as np
import tensorflow as tf
import os
import time
import pickle
import sys

from utils.general import Progbar, init_generator
from utils.parser import minibatches

class Config(object):
  n_cells = 40      # number cells units in RNN layer
                    # passed into rnn.GRUCell() or rnn. LSTMCell
  enc_seq_len = 7       #theoretically, not needed with dynamic RNN
  dec_seq_len = 8           # purposely different from enc to easily distinguish
  vocab_size = 26       # 26 letters of the alphabet and 1 for padding
  embed_size = 26
  # dropout = 0.5
  # batch_size = 202
  n_epochs = 10
  learning_rate = 0.001
  initializer = "glorot" # for xavier or "normal" for truncated normal

class Seq2SeqModel(object):
  def add_placeholders(self):
    # (batch_size, sequence_length, embedding_dimension)
    self.input_placeholder = tf.placeholder(tf.float32,
        shape=(50, 7, self.embed_size))
    # (batch_size, sequence_length, vocab_size)
    self.output_placeholder = tf.placeholder(tf.float32,
        shape=(50, 8, self.vocab_size))
    # self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

  def create_feed_dict(self, input_batch, output_batch=None):
    feed_dict = {
      self.input_placeholder: input_batch,
      self.output_placeholder: output_batch,
    }    # self.dropout_placeholder: dropout

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
    init_state = tf.get_variable('init_state', [50, self.n_cells],
         initializer=tf.contrib.layers.xavier_initializer())
    enc_seq_len = [np.sum(sen) for sen in self.questions]
    dec_seq_len = [np.sum(sen) for sen in self.answers]

    with tf.variable_scope("seq2seq") as scope:
      enc_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      dec_cell = tf.contrib.rnn.GRUCell(self.n_cells)

      # Encoder (sequence_length )
      _, enc_state = tf.nn.dynamic_rnn(enc_cell,
          self.input_placeholder, sequence_length=enc_seq_len,
          initial_state=init_state, dtype=tf.float32)
      # Intermediate decoder function
      decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
      # Decoder
      with tf.variable_scope("decoder"):
        logits, dec_state, final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, decoder_fn=decoder_fn,
            inputs=self.output_placeholder, sequence_length=8)
      with tf.variable_scope("decoder", reuse=True):
        test_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, decoder_fn=decoder_fn,
            inputs=self.output_placeholder, sequence_length=8)

    return logits, dec_state, final_context

  def add_loss_op(self, logits):
    """Adds Ops for the loss function to the computational graph.
    Args:
        pred: A tensor of shape (batch_size, n_classes)
    Returns:
        loss: A 0-d tensor (scalar) output
    """
    # loss = tf.nn.sequence_loss(logits, self.output_placeholder, weights)
    # (logits, targets, weights, average_across_timesteps=True,
    #   average_across_batch=True, softmax_loss_function=None, name=None):


    idx = tf.range(50)*tf.shape(logits)[1] + (self.dec_seq_len - 1)
    last_output = tf.gather(tf.reshape(logits, [-1, self.n_cells]), idx)

    # last_output = tf.gather_nd(logits,
    #     tf.pack([tf.range(50), self.dec_seq_len-1], axis=1))
    # logits of shape [batch_size, num_classes]
    # labels of shape [batch_size].
    dec_labels = [int(np.sum(sen)) for sen in self.answers]

    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=dec_labels, logits=last_output)
    loss = tf.reduce_mean(cross_entropy_loss)
    return loss

  def add_training_op(self, loss):
    """Sets up the training Ops.
    Args: loss: Loss tensor, from cross_entropy_loss.
    Returns: train_op: The Op for training.
    """
    optimizer = tf.train.AdamOptimizer(self.lr)
    train_op = optimizer.minimize(loss)
    return train_op

  def predict_on_batch(self, sess, inputs_batch):
    """Make predictions for the provided batch of data

    Args:
        sess: tf.Session()
        input_batch: np.ndarray of shape (n_samples, n_features)
    Returns:
        predictions: np.ndarray of shape (n_samples, n_classes)
    """
    feed = self.create_feed_dict(inputs_batch)
    predictions = sess.run(self.pred, feed_dict=feed)
    return predictions

  def build(self):
    self.add_placeholders()
    self.pred, self.dec_state, self.final_context = self.encoder_decoder()
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)

  def __init__(self, config, training_data):
    self.n_cells = config.n_cells
    self.enc_seq_len = config.enc_seq_len
    self.dec_seq_len = config.dec_seq_len
    self.embed_size = config.embed_size
    self.vocab_size = config.vocab_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.initializer = config.initializer

    # self.pretrained_embeddings = pretrained_embeddings
    self.questions = np.asarray(training_data[:50])
    self.answers = np.asarray(training_data[50:])
    # self.n_examples = training_data["questions"].shape[0]
    self.build()

def main(debug=True):
  config = Config()
  training_data = pickle.load(open("toy_data/toy_embeddings.pkl", "rb"))

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    model = Seq2SeqModel(config, training_data)
    print "took {:.2f} seconds\n".format(time.time() - start)

    init = tf.global_variables_initializer()
    # If you are using an old version of TensorFlow, you may have to use
    # this initializer instead.
    # init = tf.initialize_all_variables()
    # saver = None if debug else tf.train.Saver()

    with tf.Session() as session:
      session.run(init)

      print 80 * "="
      print "TRAINING"
      print 80 * "="
      for epoch in range(model.n_epochs):
        print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)

        fetches = [model.train_op, model.loss]    # array of desired outputs
        feed_dict = model.create_feed_dict(model.questions, model.answers)     # dictionary of inputs
        _, loss = session.run(fetches, feed_dict)

        # prog = Progbar(target=1 + model.batch_size / 50)
        # prog.update(i + 1, [("train loss", loss)])
        print loss

if __name__ == '__main__':
    main()