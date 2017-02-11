import numpy as np
import tensorflow as tf
import os
import time

from utils.general import Progbar, init_generator
from utils.parser import minibatches, loader

try:
  from tensorflow.python.ops.rnn import dynamic_rnn
except ImportError:
  from tensorflow.nn.rnn import dynamic_rnn

class Config(object):
  enc_len = 7
  dec_len = 8           # purposely different from enc to easily distinguish
  vocabulary = 27       # 26 letters of the alphabet and 1 for padding
  embed_size = 50
  hidden_size = 200
  # dropout = 0.5
  # batch_size = 2048
  n_epochs = 10
  learning_rate = 0.001
  initializer = "glorot" # for xavier or "normal" for truncated normal

class Seq2SeqModel(object):
  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, shape=(None, n_features))
    self.labels_placeholder = tf.placeholder(tf.float32, shape=(None, n_classes))
    # self.dropout_placeholder = tf.placeholder(tf.float32, shape=())

  def create_feed_dict(self, inputs_batch, labels_batch=None):
    feed_dict = {
      self.input_placeholder: input_batch,
      self.output_placeholder: output_batch,
    }    # self.dropout_placeholder: dropout

  def add_feedforward(self):
    # weight_initializer = init_generator(self.initializer)
    init_layer = tf.Variable(tf.contrib.layers.xavier_initializer)

    with tf.variable_scope("seq2seq") as scope:
      enc_cell = tf.nn.rnn_cell.GRUCell(self.enc_len)
      dec_cell = tf.nn.rnn_cell.GRUCell(self.dec_len)

      # _ refers to outputs, which are not useful in this model
      _, enc_state = dynamic_rnn(enc_cell, inputs
          sequence_length=None, initial_state=None, dtype=None)
      simple_decoder_fn = simple_decoder_fn_train(enc_state)
      outputs, dec_state, final_context = dynamic_rnn_decoder(dec_cell,
        decoder_fn=simple_decoder_fn, inputs=None, sequence_length=None)

    return outputs

  def add_loss_op(self, pred):
    """Adds Ops for the loss function to the computational graph.
    Args:
        pred: A tensor of shape (batch_size, n_classes)
    Returns:
        loss: A 0-d tensor (scalar) output
    """
    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        pred, self.labels_placeholder)
    loss = tf.reduce_mean(total_loss)


  def add_backprop(self, loss):
    """Sets up the training Ops.

    Creates an optimizer and applies the gradients to all trainable variables.
    The Op returned by this function is what must be passed to the
    sess.run() to train the model. See

    https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

    for more information.

    Args:
        loss: Loss tensor (a scalar).
    Returns:
        train_op: The Op for training.
    """

    return optimizer

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
    self.yhat = self.add_feedforward()
    self.loss = self.add_loss_op(self.yhat)
    self.train_op = self.add_backprop(self.loss)

  def fit(self, sess, saver, parser, train_examples, dev_set):
    for epoch in range(self.config.n_epochs):
      print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)
      fetches = [self.train_op, self.loss]    # array of desired outputs
      feed_dict = self.create_feed_dict(y, x)     # dictionary of inputs
      _, loss = sess.run(fetches, feed_dict)

      prog.update(i + 1, [("train loss", loss)])

      print "Evaluating on dev set",
      dev_UAS, _ = parser.parse(dev_set)
      print "- dev UAS: {:.2f}".format(dev_UAS * 100.0)

  def __init__(self, config, pretrained_embeddings):
    self.enc_len = config.enc_len
    self.dec_len = config.dec_len
    self.n_classes = config.vocabulary
    self.embed_size = config.embed_size
    self.hidden_size = config.hidden_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.initializer = config.initializer

    self.pretrained_embeddings = pretrained_embeddings
    self.build()

def main(debug=False):
  print 80 * "="
  print "INITIALIZING"
  print 80 * "="
  config = Config()
  parser, embeddings, train_examples, dev_set, test_set = load_and_preprocess_data(debug)
  if not os.path.exists('./data/weights/'):
      os.makedirs('./data/weights/')

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    model = ParserModel(config, embeddings)
    parser.model = model
    print "took {:.2f} seconds\n".format(time.time() - start)

    init = tf.global_variables_initializer()
    # If you are using an old version of TensorFlow, you may have to use
    # this initializer instead.
    # init = tf.initialize_all_variables()
    saver = None if debug else tf.train.Saver()

    with tf.Session() as session:
      parser.session = session
      session.run(init)

      print 80 * "="
      print "TRAINING"
      print 80 * "="
      model.fit(session, saver, parser, train_examples, dev_set)

      if not debug:
        print 80 * "="
        print "TESTING"
        print 80 * "="
        print "Restoring the best model weights found on the dev set"
        saver.restore(session, './data/weights/parser.weights')
        print "Final evaluation on test set",
        UAS, dependencies = parser.parse(test_set)
        print "- test UAS: {:.2f}".format(UAS * 100.0)
        print "Writing predictions"
        with open('q2_test.predicted.pkl', 'w') as f:
            cPickle.dump(dependencies, f, -1)
        print "Done!"

if __name__ == '__main__':
    main()


