import numpy as np
import tensorflow as tf
import os
import time
import pickle
import sys

from utils.general import Progbar, print_bar
from utils.parser import minibatches
from utils.getEmbeddings import get_batches, loader, embedding_to_text
from tensorflow.contrib import seq2seq

toy = False
class Config(object):
  use_attention = False
  if toy:
    n_cells = 150      # number cells units in RNN layer passed into rnn.GRUCell()
    max_enc_len = 7       # theoretically, not needed with dynamic RNN
    max_dec_len = 10       # purposely different from enc to easily distinguish
    vocab_size = 29      # 26 letters of the alphabet
    embed_size = 29      # +1 for padding, + 1 for <EOW>
    dropout_rate = 0.8
    n_epochs = 321
    learning_rate = 0.001
    batch_size = 10
  else:
    n_cells = 150
    max_enc_len = -1 # will be replaced
    max_dec_len = -1 # will be replaced
    vocab_size = -1  # will be replaced
    embed_size = 50
    dropout_rate = 0.9
    n_epochs = 300
    learning_rate = 0.001
    batch_size = 64

class Seq2SeqModel(object):
  def add_placeholders(self):

    if toy:
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
    else:
      self.question_ids = tf.placeholder(tf.int32, name='question_ids',
          shape=(self.batch_size, self.max_enc_len) )
      self.answer_ids = tf.placeholder(tf.int32, name='answer_ids',
          shape=(self.batch_size, self.max_dec_len) )
      self.enc_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
      self.dec_seq_len = tf.placeholder(tf.int32, shape=(self.batch_size,))
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


  def create_feed_dict_embeddings(self, questions_labels, answers_labels, sequence_length):
    if answers_labels is None:
      # output_data = np.zeros((self.batch_size, self.max_dec_len, self.vocab_size))
      answers_labels = np.zeros((self.batch_size,self.max_dec_len))

    feed_dict = {
      self.question_ids: questions_labels,
      self.answer_ids: answers_labels,
      self.enc_seq_len: sequence_length["enc"],
      self.dec_seq_len: sequence_length["dec"],
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

      if not toy:
        questions_batch_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.question_ids)

      with tf.variable_scope("encoder"):
        if toy:
          _, enc_state = tf.nn.dynamic_rnn(enc_cell,
              self.input_placeholder, sequence_length=self.enc_seq_len,
              initial_state=enc_init, dtype=tf.float32)
        else:
          _, enc_state = tf.nn.dynamic_rnn(enc_cell,
              questions_batch_embedding, sequence_length=self.enc_seq_len,
              initial_state=enc_init, dtype=tf.float32)

      # Decoder
      if toy:
        dec_stage = "inference" if self.labels is None else "training"
      else:
        dec_stage = "inference" if self.answer_ids is None else "training" # Weird line...
      dec_cell = tf.contrib.rnn.GRUCell(self.n_cells)
      dec_function = self.decoder_components(dec_stage, "function", enc_state)
      dec_inputs = self.decoder_components(dec_stage, "inputs", None)
      dec_seq_len = self.decoder_components(dec_stage, "sequence_length", None)
      with tf.variable_scope("decoder"):
        pred, _, _ = seq2seq.dynamic_rnn_decoder(dec_cell, dec_function,
          inputs=dec_inputs, sequence_length=dec_seq_len)
      with tf.variable_scope("decoder", reuse=True):
        pred, dec_state, _ = seq2seq.dynamic_rnn_decoder(dec_cell, dec_function,
          inputs=dec_inputs, sequence_length=dec_seq_len)

    return pred, dec_stage # not state

  """ Args:
    attention_states: hidden states to attend over
    attention_option: "luong" or "bahdanau"
    num_units, reuse: whether to reuse variable scope.
  Returns:
    attention_keys: to be compared with target states.
    attention_values: to be used to construct context vectors.
    attention_score_fn: to compute similarity between key and target states.
    attention_construct_fn: to build attention states.

    output_fn, encoder_state, attention_keys, attention_values,
    attention_score_fn, attention_construct_fn, embeddings,
    start_of_sequence_id, end_of_sequence_id, maximum_length,
    num_decoder_symbols
  """
  def decoder_components(self, stage, component_name, enc_state):

    if stage is "training":

      if toy:
        components = { "inputs": self.output_placeholder,
          "sequence_length": self.dec_seq_len }
      else:
        answers_batch_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.answer_ids)
        components = { "inputs": answers_batch_embedding,
          "sequence_length": self.dec_seq_len }
      if self.use_attention:
        keys, values, score_fn, construct_fn = prepare_attention(None,
            attention_option = "luong", num_units=self.n_cells, reuse=False)
        components["function"] = attention_decoder_fn_train(enc_state, keys,
            values, score_fn, construct_fn)
      else:
        components["function"] = seq2seq.simple_decoder_fn_train(enc_state)

    elif stage is "inference":
      output_fn, SOS_id, EOS_id = None, self.SOS_id, self.EOS_id
      components = { "inputs": None, "sequence_length": None }
      if self.use_attention:
        keys, values, score_fn, construct_fn = prepare_attention(None,
            attention_option = "luong", num_units=self.n_cells, reuse=False)
        components["function"] = attention_decoder_fn_inference(output_fn,
            enc_state, keys, values, score_fn, construct_fn, self.embedding_matrix,
            SOS_id, EOS_id, self.max_dec_len, self.vocab_size)
      else:
        components["function"] = seq2seq.simple_decoder_fn_inference(output_fn,
            enc_state, self.embedding_matrix, SOS_id, EOS_id,
              maximum_length=self.max_dec_len, num_decoder_symbols=self.vocab_size)

    return components[component_name]

  def add_loss_op(self, pred, stage):
    # for some reason, when a particular batch has sequence length less
    # than the max, the prediction are truncated to the max of that batch
    # rather than maintainingthe same length, so we
    # add in those as zeros appended to the end of the tensor
    diff = self.max_dec_len - tf.shape(pred)[1]
    diff = tf.Print(diff, [diff])
    # paddings is [   [dim1 before, dim1 after],
    #                 [dim2 before, dim2 after],
    #                 [dim3 before, dim3 after]   ]
    paddings = [[0,0], [0,diff], [0,0]]
    pred = tf.pad(pred, paddings, mode='CONSTANT', name="pad")

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
      if stage is "inference":
        final_output = tf.nn.softmax(logits)
      else:
        final_output = [2,3,4]

    final_output = tf.Print(final_output, [tf.shape(final_output)],
        first_n=3, message="Final output shape")

    # tf.summary.histogram("WeightLossy",weight)
    # tf.summary.histogram("biasLossy",bias)
    # tf.summary.histogram("logitsLossy", logits)

    # we don't pass in the final_output here since the loss function
    # already includes the softmax calculation inside
    if toy:
      cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.labels, logits=logits)
    else:
      cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.answer_ids, logits=logits)
    tf.summary.histogram("cross_entropy_loss", cross_entropy_loss)
    loss = tf.reduce_mean(cross_entropy_loss)
    return loss, final_output

  def add_training_op(self, loss):
    optimizer = tf.train.AdamOptimizer(self.lr)
    train_op = optimizer.minimize(loss)
    tf.summary.scalar("loss", loss)
    return train_op

  def predict(self, sess, test_samples, lookup):
    seq_len={"enc": get_sequence_length(test_samples),
        "dec": [self.max_dec_len for sen in test_samples]}
    _, final_output = sess.run([self.loss, self.final_output],
        self.create_feed_dict(test_samples, None, None, seq_len) )
    embedding_to_text(test_samples, final_output, lookup)

  def predict_with_embeddings(self, sess, sampleValidationQuestions, lookup, sampleValidationAnswers):
    seq_len={"enc": [len(s) for s in sampleValidationQuestions],
        "dec": [self.max_dec_len for s in sampleValidationQuestions]}
    sampleValidationQuestions = [self.addPaddingEnc(q) for q in sampleValidationQuestions]
    sampleValidationAnswers = [self.addPaddingDec(a) for a in sampleValidationAnswers]
    feed_dict = self.create_feed_dict_embeddings(sampleValidationQuestions, None, seq_len)
    _, final_output = sess.run([self.loss, self.final_output], feed_dict)
    self.word_embedding_to_text(sampleValidationQuestions, final_output, lookup, sampleValidationAnswers)

  def train(self, sess, summary_op):
    allBatches = get_batches(self.all_data, self.batch_size, True, toy)
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
        feed_dict = self.create_feed_dict(questions, answers, labels, seq_len)
      else:
        questions_labels, answers_labels = batch[0], batch[1]
        seq_len = {"enc": [len(q) for q in questions_labels], "dec": [len(a) for a in answers_labels]}
        # Pad them to be of particular size.
        questions_labels = [self.addPaddingEnc(q) for q in questions_labels]
        answers_labels = [self.addPaddingDec(a) for a in answers_labels]
        feed_dict = self.create_feed_dict_embeddings(questions_labels, answers_labels, seq_len)

      _, loss, summary = sess.run(fetches, feed_dict)
      prog.update(i + 1, [("train loss", loss)])
    # return summary

  def addPaddingEnc(self, q):
    assert self.max_enc_len >= len(q)+2, '%s,%s. Error somewhere in preparation of data. '%(q, self.max_enc_len)
    paddedQ = np.lib.pad(q, (1,self.max_enc_len+1-(len(q)+2)), 'constant', constant_values=(self.SOS_id, self.EOS_id))
    return paddedQ
  def addPaddingDec(self, a):
    assert self.max_dec_len >= len(a)+2, '%s,%s. Error somewhere in preparation of data.'%(a, self.max_dec_len)
    paddedA = np.lib.pad(a, (1,self.max_dec_len+1-(len(a)+2)), 'constant', constant_values=(self.SOS_id, self.EOS_id))
    return paddedA

  def word_embedding_to_text(self, sampleValidationQuestions, final_output, lookup, sampleValidationAnswers):
    # Final output            --> [batch_size, max_dec_len, vocab_size]
    # sampleValidationAnswers --> [batch_size, max_dec_len]
    # sampleValidationQuestions-> [batch_size, max_enc_len]
    # lookup                  --> [vocab_size]

    for b in range(self.batch_size):
      questionAsked = []
      for w in range(self.max_enc_len):
        questionAsked.append(lookup[sampleValidationQuestions[b][w]])
      expectedAnswer = []
      for w in range(self.max_dec_len):
        expectedAnswer.append(lookup[sampleValidationAnswers[b][w]]) 
      givenAnswer = []
      for w in range(self.max_dec_len):
        maxConf = max(final_output[b][w])
        if maxConf > 0.001:
          givenAnswer.append(lookup[list(final_output[b][w]).index(maxConf)])
        else:
          givenAnswer.append('--')
      print('\n')
      print("questionAsked: %s" % questionAsked)
      print("expectedAnswer: %s" % expectedAnswer)
      print("givenAnswer: %s" % givenAnswer)
      print('\n')


  def build(self):
    print("Beginning build()")
    self.add_placeholders()
    self.pred, self.stage = self.encoder_decoder()
    self.loss, self.final_output = self.add_loss_op(self.pred, self.stage)
    self.train_op = self.add_training_op(self.loss)
    print("Finished build()")

  def __init__(self, config, training_data, embedding_matrix=None):
    self.n_cells = config.n_cells
    self.embed_size = config.embed_size
    self.n_epochs = config.n_epochs
    self.lr = config.learning_rate
    self.dropout_rate = config.dropout_rate
    self.batch_size = config.batch_size
    self.use_attention = config.use_attention

    self.all_data = training_data
    self.initializer = tf.contrib.layers.xavier_initializer

    self.max_enc_len = config.max_enc_len
    self.max_dec_len = config.max_dec_len
    self.vocab_size = config.vocab_size
    if toy:
      self.SOS_id = 27
      self.EOS_id = 28
    else:
      self.embedding_matrix = tf.constant(embedding_matrix, tf.float32)
      self.SOS_id = 1
      self.EOS_id = 2
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
    validation_data = [training_data[i] for i in test_indices]
    lookup = list(' abcdefghijklmnopqrstuvwxyz ')
    statistics = {"embedding_matrix": np.eye(config.vocab_size)}
    embedding_matrix = None
  else:
    loadedData = loader()
    training_data = loadedData["training_data"]
    validation_data = loadedData["validation_data"]
    config.vocab_size = loadedData["vocab_size"]
    config.max_dec_len = loadedData["max_dec_len"]
    config.max_enc_len = loadedData["max_enc_len"]
    embedding_matrix = loadedData["embedding_matrix"]
    lookup = loadedData["vocabs_list"]

  with tf.Graph().as_default():
    print "Building model...",
    start = time.time()
    model = Seq2SeqModel(config, training_data, embedding_matrix)
    print "Model Built! Took {:.2f} seconds\n".format(time.time() - start)

    with tf.Session() as session:
      varInitializer = tf.global_variables_initializer()
      session.run(varInitializer)

      summary_op = tf.summary.merge_all()
      # saver = None if debug else tf.train.Saver()

      print_bar("training")
      for epoch in range(model.n_epochs):
        model.train(session, summary_op)

        if epoch%40 == 0 and toy:
          print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)
          # print_bar("prediction")
          test_indices = np.random.choice(50, 10, replace=False)
          validation_data = [training_data[i] for i in test_indices]
          predictions = model.predict(session, validation_data, lookup)
        if epoch%1 == 0 and not toy:
          print "Epoch {:} out of {:}".format(epoch + 1, model.n_epochs)
          # print_bar("prediction")
          test_indices = np.random.choice(len(validation_data[0]), config.batch_size, replace=False)
          sampleValidationQuestions = [validation_data[0][i] for i in test_indices]
          sampleValidationAnswers = [validation_data[1][i] for i in test_indices]
          predictions = model.predict_with_embeddings(session, sampleValidationQuestions, lookup, sampleValidationAnswers)


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
