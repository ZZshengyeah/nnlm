'''
# @Time    : 18-9-7 下午1:49
# @Author  : ShengZ
# @FileName: lstm_model.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

from preprocess import *
from load_model import BaseModel
FLAGS = tf.app.flags.FLAGS


def fully_connect_layer(input_tensor, hidden_size, output_size):
    with tf.variable_scope('fully_connect'):
        weights = tf.get_variable('weight',
                                  [hidden_size,output_size],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  trainable=True)
        biases = tf.get_variable('bias',[output_size],initializer=tf.constant_initializer(0.1),trainable=True)
        loss_l2 = tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        return tf.matmul(input_tensor,weights) + biases, loss_l2

class lstm_model(BaseModel):
    def __init__(self, is_training, batch_size, num_steps):

        self.batch_size = batch_size
        self.num_steps = num_steps

        self.input_data = tf.placeholder(tf.int32, [batch_size,num_steps])
        self.label = tf.placeholder(tf.int32, [batch_size,num_steps])

        keep_prob = FLAGS.keep_prob if is_training else 1.0


        lstm_cells = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(FLAGS.hidden_size,forget_bias=1.0),
                                                                                    output_keep_prob=keep_prob)
                                                                                    for _ in range(FLAGS.num_layers)]
        cells = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        self.initial_state = cells.zero_state(batch_size,tf.float32)
        trimed_embed, _ = maybe_trim_embedding(FLAGS.vocab_file,
                                             FLAGS.pretrain_embed_file,
                                             FLAGS.pretrain_vocab_file,
                                             FLAGS.trimed_embed_file)
        embedding = tf.get_variable('embedding',initializer=trimed_embed)
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        '''
        input_data.shape = [batch_size,num_steps]
        inputs.shape = [batch_size, num_steps, word_dim]
        '''
        outputs = []
        state = self.initial_state
        with tf.variable_scope('lstm'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cells(inputs[:, time_step, :], state)
                cell_output = tf.reshape(cell_output,[-1,FLAGS.hidden_size])
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs,axis=1), [-1, FLAGS.hidden_size])

        if is_training:
            output = tf.nn.dropout(output,keep_prob=keep_prob)
        logits, loss_l2 = fully_connect_layer(output, FLAGS.hidden_size, FLAGS.vocab_size)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.label,[-1]))

        self.losses = tf.reduce_sum(losses) / batch_size
        self.perp = tf.exp(self.losses / self.num_steps)
        self.final_state = state

        if not is_training:
            return

        trainable_varibales = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.losses, trainable_varibales), FLAGS.max_grad_norm)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads,trainable_varibales))




