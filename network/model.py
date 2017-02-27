import numpy as np
import tensorflow as tf
import os
import time
import pickle
import sys

from utils.general import Progbar, print_bar
from utils.parser import minibatches
from utils.getEmbeddings import get_batches, loader, embedding_to_text

toy = True
class Config(object):
  if toy:
    n_cells = 90      # number cells units in RNN layer passed into rnn.GRUCell()
    max_enc_len = 7       # theoretically, not needed with dynamic RNN
    max_dec_len = 10       # purposely different from enc to easily distinguish
    vocab_size = 29      # 26 letters of the alphabet
    embed_size = 29      # +1 for padding, + 1 for <EOW>
    dropout_rate = 1.0
    n_epochs = 30
    learning_rate = 0.001
    batch_size = 10
  else:
    n_cells = 150
    max_enc_len = 100
    max_dec_len = 100
    vocab_size = 50000  # to be replaced
    embed_size = 300
    dropout_rate = 0.9
    n_epochs = 3
    learning_rate = 0.001
    batch_size = 64

class Seq2SeqModel(object):
  def add_placeholders(self):
    # (batch_size, sequence_length, embedding_dimension)
    self.input_placeholder = tf.placeholder(tf.float32, name='question',
        shape=(self.batch_size, self.max_enc_len, self.embed_size) )
    # (batch_size, sequence_length, vocab_size)
    self.output_placeholder = tf.placeholder(tf.float32, name='answer',
        shape=(self.batch_size, self.max_dec_len, self.embed_size) )
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
      output_data = np.zeros((self.batch_size, self.max_dec_len, self.vocab_size))
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

    with tf.variable_scope("seq2seq") as scope:
      # Encoder
      enc_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      enc_cell = tf.contrib.rnn.DropoutWrapper(cell=enc_cell,
        output_keep_prob=self.dropout_rate)  # Important: RNN version of Dropout!
      enc_init = tf.get_variable('init_state', [self.batch_size, self.n_cells],
         initializer=self.initializer())
      with tf.variable_scope("encoder"):
        _, enc_state = tf.nn.dynamic_rnn(enc_cell,
            self.input_placeholder, sequence_length=self.enc_seq_len,
            initial_state=enc_init, dtype=tf.float32)

      # Decoder
      dec_stage = "inference" if self.labels is None else "training"
      dec_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      dec_function = self.decoder_components(dec_stage, "function", enc_state)
      dec_inputs = self.decoder_components(dec_stage, "inputs", None)
      dec_seq_len = self.decoder_components(dec_stage, "sequence_length", None)
      with tf.variable_scope("decoder"):
        pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
          dec_function, inputs=dec_inputs, sequence_length=dec_seq_len)
      with tf.variable_scope("decoder", reuse=True):
        pred, dec_state, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
          dec_function, inputs=dec_inputs, sequence_length=dec_seq_len)

    return pred, dec_state

  def decoder_components(self, stage, component_name, enc_state):
    if stage is "training":
      components = { "inputs": self.output_placeholder,
        "function": tf.contrib.seq2seq.simple_decoder_fn_train(enc_state),
        "sequence_length": self.dec_seq_len }
    elif stage is "inference":
      output_fn, SOS_id, EOS_id = None, 27, 28
      components = { "inputs": None,
        "function": tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn,
            enc_state, self.embedding_matrix, SOS_id, EOS_id,
            maximum_length=self.max_dec_len, num_decoder_symbols=self.vocab_size),
        "sequence_length": None }
    return components[component_name]

  def add_loss_op(self, pred):
    # for some reason, when a particular batch has sequence length less
    # than the max, the prediction are truncated to the max of that batch
    # rather than maintainingthe same length, so we
    # add in those as zeros appended to the end of the tensor
    # diff = self.max_dec_len - tf.shape(pred)[1]
    # diff = tf.Print(diff, [diff])
    # paddings is [   [dim1 before, dim1 after],
    #                 [dim2 before, dim2 after],
    #                 [dim3 before, dim3 after]   ]
    # paddings = [[0,0], [0,diff], [0,0]]
    # pred = tf.pad(pred, paddings, mode='CONSTANT', name="pad")

    flat_size = self.max_dec_len * self.n_cells
    flattened_preds = tf.reshape(pred, [self.batch_size, flat_size])
    # now the predictions are shape (10, 8*40)

    # logits of shape [batch_size, seq_len, vocab_size]
    # labels of shape [batch_size, seq_len].
    with tf.variable_scope('lossy', initializer=self.initializer()):
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

  def predict(self, sess, test_samples, lookup):
    if toy:
      seq_len={"enc": get_sequence_length(test_samples),
          "dec": [self.max_dec_len for sen in test_samples]}
      _, final_output = sess.run([self.loss, self.final_output],
          self.create_feed_dict(test_samples, None, None, seq_len) )
      embedding_to_text(test_samples, final_output, lookup)
    else:
      print "To be added"

  def train(self, sess, summary_op):
    allBatches = get_batches(self.all_data, self.batch_size, False, toy=True)
    prog = Progbar(target=(len(self.all_data)/2) / self.batch_size)
    fetches = [self.train_op, self.loss, summary_op]    # array of desired outputs

    for i, batch in enumerate(allBatches):
      if toy:
        questions, answers = batch[0], batch[1]
        enc_seq_len = get_sequence_length(questions)
        dec_seq_len = [self.max_dec_len for sen in answers]
        # get_sequence_length(answers)
        seq_len = {"enc": enc_seq_len, "dec": dec_seq_len}
        # print seq_len
        labels = [ [letter.index(1) for letter in word] for word in answers]
        labels = np.asarray(labels)
      else:
        questions, answers = batch["questions"], batch["answers"]
        seq_len = {"enc": batch["enc"], "dec": batch["dec"]}
        labels = batch["labels"]

      feed_dict = self.create_feed_dict(questions, answers, labels, seq_len)
      _, loss, summary = sess.run(fetches, feed_dict)
      prog.update(i + 1, [("train loss", loss)])
    # return summary

  def build(self):
    self.add_placeholders()
    self.pred, self.dec_state = self.encoder_decoder()
    self.loss, self.final_output = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)

  def __init__(self, config, training_data, statistics):
    self.n_cells = config.n_cells
    self.embed_size = config.embed_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.dropout_rate = config.dropout_rate
    self.batch_size = config.batch_size
    self.all_data = training_data
    self.initializer = tf.contrib.layers.xavier_initializer

    if toy:
      self.max_enc_len = config.max_enc_len
      self.max_dec_len = config.max_dec_len
      self.vocab_size = config.vocab_size
    else:
      self.max_enc_len = statistics.max_enc_len
      self.max_dec_len = statistics.max_dec_len
      self.vocab_size = statistics.vocab_size
    self.embedding_matrix = statistics["embedding_matrix"]

    self.build()

def get_sequence_length(batch):
  def word_seq_len(word):
    letter_sequence_lengths = map(lambda letter: sum(letter[1:]), word)
    return int(sum(letter_sequence_lengths))
  return map(word_seq_len, batch)

def main(debug=True):
  config = Config()
  if toy:
    training_data = pickle.load(open("dirty/toy_data/toy_embeddings.pkl", "rb"))
    test_indices = np.random.choice(50, 10, replace=False)
    # test_indices = range(30,40)
    test_data = [training_data[i] for i in test_indices]
    lookup = list(' abcdefghijklmnopqrstuvwxyz |')
    statistics = {"embedding_matrix": np.eye(config.vocab_size)}
  else:
    all_data = utils.loader(False)
    training_data = all_data["training_data"]
    test_data = all_data["test_data"]
    statistics = all_data["statistics"]
    lookup = None

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    model = Seq2SeqModel(config, training_data, statistics)
    print "Model Built! Took {:.2f} seconds\n".format(time.time() - start)

    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      summary_op = tf.summary.merge_all()
      # saver = None if debug else tf.train.Saver()

      print_bar("training")
      for epoch in range(model.n_epochs):
        print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)
        model.train(session, summary_op)

        # if epoch%40 == 0:
      print_bar("prediction")
      predictions = model.predict(session, test_data, lookup)

if __name__ == '__main__':
    # training_data = pickle.load(open("dirty/toy_data/toy_embeddings_new.pkl", "rb"))
    # test_indices = np.random.choice(50, 10, replace=False)
    # lookup = list(' abcdefghijklmnopqrstuvwxyz ')
    # test_data = [training_data[i] for i in test_indices]
    # answer_data = [training_data[i+50] for i in test_indices]
    # embedding_to_text(test_data, answer_data, lookup)
    main()

    # logs_path = '/tmp/tensorflow/board'
    # writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    # Tensorboard. Run: tensorboard --logdir=run1:/tmp/tensorflow/board --port 6006
    # writer.add_summary(summary, epoch)# * batch_count + i)

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
