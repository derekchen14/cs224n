import numpy as np
import tensorflow as tf
import os
import time
import pickle
import sys

from utils.general import Progbar, init_generator
from utils.parser import minibatches
from utils.getEmbeddings import get_minibatches, loader

class Config(object):
  n_cells = 200      # number cells units in RNN layer
                    # passed into rnn.GRUCell() or rnn. LSTMCell
  embed_size = 300
  # dropout = 0.5
  # batch_size = 202
  n_epochs = 10
  learning_rate = 0.001
  initializer = "glorot" # for xavier or "normal" for truncated normal

  # used by loader.
  data_path = './clean'
  minibatch_size = 20
  reduced_size = 200
  reduced_train = 120
  reduced_dev = 60
  reduced_test = 20
  test_size = 0.8
  dev_size = 0.15
  test_size = 1.0 - test_size - dev_size

class Seq2SeqModel(object):
  def add_placeholders(self):
    # (batch_size, sequence_length, embedding_dimension)
    self.input_placeholder = tf.placeholder(tf.int32,
        shape=(self.batch_size, None))
    # (batch_size, sequence_length, vocab_size)
    self.output_placeholder = tf.placeholder(tf.int32,
        shape=(self.batch_size, None))
    self.enc_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
    self.dec_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
    # self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

  def create_feed_dict(self, input_batch, output_batch, sequence_length):
    feed_dict = {
      self.input_placeholder: input_batch,
      self.output_placeholder: output_batch,
      self.enc_seq_len: sequence_length["enc"],
      self.dec_seq_len: sequence_length["dec"]
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

  def add_embedding(self):
    # Maybe one day change to Variable.
    embedding_tensor = tf.Variable(self.pretrained_embeddings)
    questions = tf.nn.embedding_lookup(embedding_tensor, self.input_placeholder)
    answers = tf.nn.embedding_lookup(embedding_tensor, self.output_placeholder)
    return questions, answers

  def encoder_decoder(self):
    init_state = tf.get_variable('init_state', [self.batch_size, self.n_cells],
         initializer=tf.contrib.layers.xavier_initializer())

    questions, answers = self.add_embedding()
    # print questions.get_shape()
    # print answers.get_shape()
    # (20, ?, 300)
    # (20, ?, 300)
    # This tuple represents a random Q/A pair, we will check it makes any sense
    pair = (questions[3], answers[3])
    self.embedding_to_text(pair)

    with tf.variable_scope("seq2seq") as scope:
      enc_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      dec_cell = tf.contrib.rnn.GRUCell(self.n_cells)

      # Encoder (sequence_length )
      _, enc_state = tf.nn.dynamic_rnn(enc_cell,
          questions, sequence_length=self.enc_seq_len,
          initial_state=init_state, dtype=tf.float32)
      # Intermediate decoder function
      decoder_fn = tf.contrib.seq2seq.simple_decoder_fn_train(enc_state)
      # Decoder
      with tf.variable_scope("decoder"):
        logits, dec_state, final_context = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, decoder_fn=decoder_fn, inputs=answers,
            sequence_length=tf.reduce_max(self.dec_seq_len))
      with tf.variable_scope("decoder", reuse=True):
        test_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            dec_cell, decoder_fn=decoder_fn, inputs=answers,
            sequence_length=tf.reduce_max(self.dec_seq_len))

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

    print logits.get_shape()
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,self.labels_placeholder))

    idx = tf.range(self.batch_size)*tf.shape(logits)[1] + (self.dec_seq_len - 1)
    last_output = tf.gather(tf.reshape(logits, [-1, self.n_cells]), idx)

    # last_output = tf.gather_nd(logits,
    #     tf.pack([tf.range(50), self.dec_seq_len-1], axis=1))
    # logits of shape [batch_size, num_classes]
    # labels of shape [batch_size].
    dec_labels = [int(np.sum(sen)) for sen in self.answers]

    # might be sparse
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
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

  def train_on_batch(self, batch, sess):
    fetches = [self.train_op, self.loss]    # array of desired outputs
    answers = batch[0]
    questions = batch[1]
    sequence_length = {
      "enc": [len(a) for a in answers],
      "dec":  [len(q) for q in questions]
    }
    feed_dict = self.create_feed_dict(questions, answers, sequence_length)     # dictionary of inputs
    _, loss = sess.run(fetches, feed_dict)
    return loss

  def embedding_to_text(self, pair):
    q, a = pair
    x = tf.Print(q, [q], message="I wrote this ---------")
    # sess = tf.InteractiveSession()
    # sess.run(y)
    # print q.eval(session=sess)
    # sys.exit()

  def predict_on_batch(self, sess, inputs_batch):
    """Make predictions for the provided batch of data

    Args:
        sess: tf.Session()
        input_batch: np.ndarray of shape (n_samples, n_features)
    Returns:
        predictions: np.ndarray of shape (n_samples, n_classes)
    """
    # decoder
    feed = self.create_feed_dict(inputs_batch)
    predictions = sess.run(self.pred, feed_dict=feed)
    return predictions

  def build(self):
    self.add_placeholders()
    self.pred, self.dec_state, self.final_context = self.encoder_decoder()
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)

  def __init__(self, config, embedding_matrix):
    self.n_cells = config.n_cells
    self.embed_size = config.embed_size
    self.vocab_size = len(embedding_matrix)
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.initializer = config.initializer

    self.batch_size = config.minibatch_size
    # self.questions = training_data[1]
    # self.answers = np.asarray(training_data[:50])
    self.pretrained_embeddings = embedding_matrix
    # self.n_examples = training_data["questions"].shape[0]
    self.build()

def main(debug=True):
  config = Config()
  # training_data = pickle.load(open("toy_data/toy_embeddings.pkl", "rb"))
  train, dev, _ , embedding_matrix, _ = loader(config, reduced=True)

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    allData = train
    model = Seq2SeqModel(config, embedding_matrix)
    print "took {:.2f} seconds\n".format(time.time() - start)

    init = tf.global_variables_initializer()
    # saver = None if debug else tf.train.Saver()

    with tf.Session() as session:
      session.run(init)

      print 80 * "="
      print "TRAINING"
      print 80 * "="

      for epoch in range(model.n_epochs):
        print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)

        allBatches = get_minibatches(allData, config.minibatch_size, shuffle=False)
        data_size = len(allData[0])

        for batch in allBatches:
          loss = model.train_on_batch(batch, session)
          # prog = Progbar(target=1 + config.minibatch_size / data_size)
          # prog.update(i + 1, [("train loss", loss)])

if __name__ == '__main__':
    main()