'''
# @Time    : 18-9-6 下午6:32
# @Author  : ShengZ
# @FileName: preprocess.py
# @Software: PyCharm
# @Github  : https://github.com/ZZshengyeah
'''
import os
import codecs
import tensorflow as tf
import numpy as np

flags = tf.app.flags

### 数据文件
flags.DEFINE_string('raw_train_data','./data/ptb.train.txt','raw_traindata')
flags.DEFINE_string('raw_valid_data','./data/ptb.valid.txt','raw_validdata')
flags.DEFINE_string('raw_test_data','./data/ptb.test.txt','raw_testdata')
flags.DEFINE_string('train_data', './data/ptb.train', 'train_data_output')
flags.DEFINE_string('valid_data', './data/ptb.valid', 'valid_data_output')
flags.DEFINE_string('test_data', './data/ptb.test', 'test_data_output')
flags.DEFINE_string("logdir", "save_model/", "where to save the model")

### 数据处理
flags.DEFINE_string('vocab_file', './data/ptb.vocab','vocab_file_path')
flags.DEFINE_string('trimed_embed_file','./data/trimed_embedding.npy','trim_embedding')
flags.DEFINE_string('pretrain_embed_file','./data/glove_vector_50.npy','pretrain_embedding')
flags.DEFINE_string('pretrain_vocab_file','./data/glove_words.txt','pretrain_vocab')

### 网络参数
flags.DEFINE_integer('hidden_size',256, 'hidden_layer_nodes')
flags.DEFINE_integer('max_len', 15, 'max_length of sentence')
flags.DEFINE_integer('batch_size', 20, 'batch size')
flags.DEFINE_integer('num_steps',30 , 'number of steps')
flags.DEFINE_integer('word_dim',50,'dimention of word')
flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
flags.DEFINE_integer('max_grad_norm',5,'max grad norm')
flags.DEFINE_integer('num_epoches',200,'epoches number')
flags.DEFINE_float('keep_prob', 0.8, 'dropout possibility')
flags.DEFINE_float('learning_rate',0.001,'learning rate')
flags.DEFINE_integer('vocab_size',10000,'size of vocab')


FLAGS = tf.flags.FLAGS


def maybe_write_vocab():
    if not os.path.exists(FLAGS.vocab_file):
        print('writing vocab!')
        vocab = set()
        with codecs.open(FLAGS.raw_train_data,'r','utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    vocab.add(word)
        print(len(vocab))
        with codecs.open(FLAGS.raw_train_data,'r','utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    vocab.add(word)
        print(len(vocab))
        with codecs.open(FLAGS.raw_train_data,'r','utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    vocab.add(word)
        print(len(vocab))
        with codecs.open(FLAGS.vocab_file,'w','utf-8') as f:
            for word in list(vocab):
                f.write(word)
                f.write('\n')

def load_vocab(vocab_file):
    vocab = []
    with codecs.open(vocab_file,'r','utf-8') as f:
        for line in f:
            word = line.strip()
            vocab.append(word)
    return vocab

def load_embedding(embed_file,vocab_file):
    embed = np.load(embed_file)
    word2id = {}
    words = load_vocab(vocab_file)
    for id, w in enumerate(words):
        word2id[w] = id
    return embed, word2id

def maybe_trim_embedding(vocab_file,
                         pretrain_embed_file,
                         pretrain_vocab_file,
                         trimed_embed_file):
    if not os.path.exists(trimed_embed_file):
        print('trimming word embedding!\n')
        pretrain_embed,pretrain_word2id = load_embedding(pretrain_embed_file,
                                                         pretrain_vocab_file)
        new_word_embed = []
        vocab_list = load_vocab(vocab_file)
        for word in vocab_list:
            if word in pretrain_word2id:
                id = pretrain_word2id[word]
                new_word_embed.append(pretrain_embed[id])
            else:
                vec = np.random.normal(0,0.1,[50])
                new_word_embed.append(vec)
        new_word_embed.append(np.random.normal(0,0.1,[50]))
        new_word_embed = np.asarray(new_word_embed)
        np.save(trimed_embed_file,new_word_embed.astype(np.float32))
    new_word_embed, vocab2id = load_embedding(trimed_embed_file,vocab_file)
    print('trimed_shape:',new_word_embed.shape)
    return new_word_embed, vocab2id

def maybe_map_words_to_ids(raw_data, target_data):
    if not os.path.exists(target_data):
        with codecs.open(FLAGS.vocab_file,'r','utf-8') as f:
            vocab = [word.strip() for word in f.readlines()]
        word2id = {}
        for i, word in enumerate(vocab):
            word2id[word] = i
        with codecs.open(raw_data,'r','utf-8') as f:
            with codecs.open(target_data,'w','utf-8') as ff:
                lines =f.readlines()
                for line in lines:
                    line = line.strip().split()
                    ff_line = [str(word2id[x]) for x in line]
                    for id in ff_line:
                        ff.write(id+' ')
                    ff.write('\n')

def join_all_str(data):
    with codecs.open(data,'r','utf-8') as f:
        all_join = ' '.join([line.strip() for line in f])
    return all_join.split()

def make_batches(data, batch_size, num_steps):
    all_join_list = join_all_str(data)
    num_batches = (len(all_join_list)-1) // (batch_size * num_steps)
    data = np.array(all_join_list[:num_batches * batch_size * num_steps])
    data = np.reshape(data,[batch_size,num_batches*num_steps])
    data_batches = np.split(data,num_batches,axis=1)

    label = np.array(all_join_list[1:num_batches*batch_size * num_steps+1])
    label = np.reshape(label,[batch_size,num_batches*num_steps])
    label_batches = np.split(label,num_batches,axis=1)

    return list(zip(data_batches,label_batches))

def main(_):

    maybe_write_vocab()
    maybe_map_words_to_ids(FLAGS.raw_train_data,FLAGS.train_data)
    maybe_map_words_to_ids(FLAGS.raw_valid_data, FLAGS.valid_data)
    maybe_map_words_to_ids(FLAGS.raw_test_data, FLAGS.test_data)

    trimed_embed, vocab2id = maybe_trim_embedding(FLAGS.vocab_file,
                         FLAGS.pretrain_embed_file,
                         FLAGS.pretrain_vocab_file,
                         FLAGS.trimed_embed_file)
    print(trimed_embed)
    print(vocab2id)

if __name__ == '__main__':
    #just for test
    tf.app.run()