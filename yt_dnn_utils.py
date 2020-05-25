import argparse
import json
import collections

import tensorflow as tf

from tensorflow.contrib.layers import fully_connected, batch_norm

#### argument parser

parser = argparse.ArgumentParser(description='yt_dnn parser')
parser.add_argument('-b', dest='batch_size', type=int, default=256)
parser.add_argument('-k', dest='k', type=int, default=32)

parser.add_argument('-init_stddev', dest='init_stddev', type=float, default=0.01)

parser.add_argument('-opt', dest='opt', type=str, default="adam")
parser.add_argument('-lr_decay', dest='lr_decay', default=False, action="store_true", help="")
parser.add_argument('-eta', dest='eta', type=float, default=0.001)

parser.add_argument('-lambda_nn', dest='lambda_nn', type=float, default=0.01)
parser.add_argument('-lambda_emb', dest='lambda_emb', type=float, default=0.01)

parser.add_argument('-ema', dest='ema', default=False, action="store_true")

parser.add_argument('-id', dest='id', type=str)

parser.add_argument('-max_epoch', dest='max_epoch', type=int, default=20000)

parser.add_argument('-custom_loss', dest='custom_loss', default=False, action="store_true")

parser.add_argument('-feat_info_path', dest='feat_info_path', type=str)
parser.add_argument('-pub_id_mapping_path', dest='pub_id_mapping_path', type=str)

parser.add_argument('-train_path', dest='train_path', type=str)
parser.add_argument('-test_path', dest='test_path', type=str)
parser.add_argument('-all_path', dest='all_path', type=str)

parser.add_argument('-date', dest='date', type=str)

parser.add_argument('-for_eval', dest='for_eval', default=False, action="store_true")

#### parse feat info json

FeatInfo = collections.namedtuple('FeatInfo', 'max_feat_id dense_feat_ids slot_ids pub_id_offset pub_id_num')


def parse_feat_info(path):
    with open(path) as f:
        obj = json.load(f)
    return FeatInfo(obj["max_feat_id"], obj["dense_feat_ids"], obj["slot_ids"], obj["pub_id_offset"], obj["pub_id_num"])


####

def get_context_features(n_dense):
    print("====> get_context_features, n_dense: {0}".format(n_dense))
    return {'device_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
            'req_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
            'dense': tf.FixedLenFeature(shape=[n_dense], dtype=tf.float32),
            'pos': tf.VarLenFeature(dtype=tf.int64),
            'neg': tf.VarLenFeature(dtype=tf.int64)}


sequence_features = {'seq_idx': tf.VarLenFeature(dtype=tf.int64),
                     'seq_val': tf.VarLenFeature(dtype=tf.float32)}


def get_parse_seq_example_fn(n_dense):
    def parse_seq_example_criteo(example_proto):
        ex = tf.parse_single_sequence_example(example_proto, get_context_features(n_dense), sequence_features)
        return ex

    return parse_seq_example_criteo


def get_dataset_iterator(b_size, path, n_dense):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(get_parse_seq_example_fn(n_dense))
    dataset = dataset.repeat()
    dataset = dataset.batch(b_size)
    return dataset.make_initializable_iterator()


####

def fc(inputs, output_size, is_training, w_initializer, activation_fn, keep_prob, use_batch_norm, bn_decay, l2_scale_nn, scope):
    batch_norm_args = {}
    if use_batch_norm:
        batch_norm_args = {
            'normalizer_fn': batch_norm,
            'normalizer_params': {
                'is_training': is_training,
                'decay': bn_decay,
                'scale': True,
                'updates_collections': None
            }
        }
    with tf.variable_scope(scope):
        print("====> inputs: {0}, output_size: {1}, is_training: {2}, activation_fn: {3}, dropout_keep_prob: {4}, scope: {5}"
              .format(inputs, output_size, is_training, activation_fn, keep_prob, scope))
        outputs = fully_connected(inputs=inputs, num_outputs=output_size, activation_fn=activation_fn,
                                  weights_initializer=w_initializer,
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_scale_nn),
                                  biases_regularizer=tf.contrib.layers.l2_regularizer(l2_scale_nn),
                                  scope=scope, **batch_norm_args)
        return tf.cond(is_training, lambda: tf.nn.dropout(outputs, keep_prob=keep_prob), lambda: outputs)


def get_padded_pos_neg(y_pos, y_neg, default_value=-1):
    y_pos_ragged = tf.RaggedTensor.from_tensor(tf.sparse.to_dense(y_pos, default_value=-1), padding=-1)
    y_neg_ragged = tf.RaggedTensor.from_tensor(tf.sparse.to_dense(y_neg, default_value=-1), padding=-1)
    #
    y_id_ragged = tf.concat([y_pos_ragged, y_neg_ragged], axis=1)
    y_id_padded = y_id_ragged.to_tensor(default_value=default_value)
    #
    y_label_ragged = tf.concat([tf.ones_like(y_pos_ragged), tf.zeros_like(y_neg_ragged)], axis=1)
    y_label_padded = y_label_ragged.to_tensor(default_value=default_value)
    #
    return y_id_padded, y_label_padded, tf.reduce_min(y_id_ragged.row_lengths())


def load_pub_id_mapping(path):
    with open(path) as f:
        obj = json.load(f)
    return {v: k for k, v in obj['pub_id'].items()}
