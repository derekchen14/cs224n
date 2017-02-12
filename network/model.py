import numpy as np
import tensorflow as tf
import os
import time
import pickle

from utils.general import Progbar, init_generator
from utils.parser import minibatches

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

  def encoder_decoder(self):
    # weight_initializer = init_generator(self.initializer)
    init_layer = tf.Variable(tf.contrib.layers.xavier_initializer)

    with tf.variable_scope("seq2seq") as scope:
      enc_cell = tf.nn.rnn_cell.GRUCell(self.enc_len)
      dec_cell = tf.nn.rnn_cell.GRUCell(self.dec_len)

      # _ refers to outputs, which are not useful in this model
      _, enc_state = dynamic_rnn(enc_cell, inputs,
          sequence_length=None, initial_state=None, dtype=None)
      simple_decoder_fn = simple_decoder_fn_train(enc_state)
      outputs, dec_state, final_context = dynamic_rnn_decoder(dec_cell,
        decoder_fn=simple_decoder_fn, inputs=None, sequence_length=None)

    return outputs, dec_state, final_context

  def add_loss_op(self, pred, weights):
    """Adds Ops for the loss function to the computational graph.
    Args:
        pred: A tensor of shape (batch_size, n_classes)
    Returns:
        loss: A 0-d tensor (scalar) output
    """
    loss = tf.nn.sequence_loss( pred, self.output_placeholder, weights)
    # (logits, targets, weights, average_across_timesteps=True,
    #   average_across_batch=True, softmax_loss_function=None, name=None):
    # loss = tf.reduce_mean(total_loss)
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
    self.loss = self.add_loss_op(self.pred, self.final_context)
    self.train_op = self.add_training_op(self.loss)

  def __init__(self, config, training_data):
    self.enc_len = config.enc_len
    self.dec_len = config.dec_len
    self.n_classes = config.vocabulary
    self.embed_size = config.embed_size
    self.hidden_size = config.hidden_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.initializer = config.initializer

    # self.pretrained_embeddings = pretrained_embeddings
    self.questions = training_data["questions"]
    self.answers = training_data["answers"]
    self.n_examples = training_data["questions"].shape[0]
    self.build()

def main(debug=True):
  config = Config()
  # embeddings, train_examples, dev_set, test_set = loader
  training_data = pickle.load(open("spaCy/encode.py", "rb"))
    # loadedData = pickle.load(open(writeTo, "rb"))
  # if not os.path.exists('./data/weights/'):
  #     os.makedirs('./data/weights/')

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
      for epoch in range(self.config.n_epochs):
        print "Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs)

        fetches = [model.train_op, model.loss]    # array of desired outputs
        feed_dict = model.create_feed_dict(answers, questions)     # dictionary of inputs
        _, loss = session.run(fetches, feed_dict)

        prog = Progbar(target=1 + model.n_examples / model.batch_size)
        prog.update(i + 1, [("train loss", loss)])

if __name__ == '__main__':
    main()


