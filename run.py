'''
# @Time    : 18-9-7 下午5:42
# @Author  : ShengZ
# @FileName: run.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''

import sys
import tensorflow as tf
import numpy as np
import time
from lstm_model import lstm_model
from preprocess import make_batches

FLAGS = tf.app.flags.FLAGS

def run_epoch(session, model, batches, train_op):
    total_losses = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for x, y in batches:

        fetches = [model.losses, model.final_state, train_op]
        loss, state, _ = session.run(fetches,
                                     {model.input_data:x, model.label:y,model.initial_state:state})
        total_losses += loss
        iters += model.num_steps
        perp = np.exp(total_losses /iters)

    return perp

def train(sess):

    with tf.name_scope('Train'):
        with tf.variable_scope('language_model',reuse=None):
            train_model = lstm_model(True, FLAGS.batch_size, FLAGS.num_steps)
            train_model.set_saver('lstm-%d-%d' %(FLAGS.num_epoches,FLAGS.word_dim))
    with tf.name_scope('Test'):
        with tf.variable_scope('language_model',reuse=True):
            test_model = lstm_model(False, 1, 1)

    tf.global_variables_initializer().run()
    train_batches = make_batches(FLAGS.train_data, FLAGS.batch_size, FLAGS.num_steps)
    valid_batches = make_batches(FLAGS.valid_data, 1, 1)

    epoch = 1
    n = 1
    best_step = n
    best = .0
    start = time.time()
    origin_begin = start

    while epoch <= FLAGS.num_epoches:
        total_losses = 0.0
        iters = 0

        train_state = sess.run(train_model.initial_state)
        for x, y in train_batches:
            fetches = [train_model.losses, train_model.final_state, train_model.train_op]
            loss, state, _ = sess.run(fetches,
                                         {train_model.input_data: x, train_model.label: y,train_model.initial_state:train_state})
            total_losses += loss
            iters += train_model.num_steps

        valid_perp = 0
        test_state = sess.run(test_model.initial_state)

        for test_x, test_y in valid_batches:
            valid_perp += sess.run(test_model.perp, {test_model.input_data: test_x, test_model.label: test_y,
                                                                               test_model.initial_state:test_state})
        valid_perp = valid_perp / len(valid_batches)
        perp = np.exp(total_losses / iters)
        if best < valid_perp:
            best = valid_perp
            best_step = n
        end = time.time()
        print('----epoch: {}  perp: {}  best_perp: {}  time: {} ----'.format(epoch, perp, valid_perp, end-start))
        start = end
        epoch += 1
    duration = (time.time() - origin_begin) / 3600
    train_model.save(sess, best_step)

    print('Done training! best_step: {}, best_perp: {}'.format(best_step,best))
    print('duration: %.2f hours' %duration)



def test(sess):
    test_model = lstm_model(False, 1, 1)
    test_model.restore(sess)
    test_batches = make_batches(FLAGS.test_data, 1, 1)
    _, test_perplexity, __, ___ = run_epoch(sess, test_model, test_batches, tf.no_op(), 0, 0, )
    print('finished test, test perplexity: {}'.format(test_perplexity))

def main(_):
    goal = sys.argv[1].split('--')[1]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if goal == 'train':
                train(sess)
            elif goal == 'test':
                test(sess)

if __name__ == '__main__':
    tf.app.run()
