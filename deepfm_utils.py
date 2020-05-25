import argparse
import json
import collections

import tensorflow as tf

from tensorflow.contrib.layers import fully_connected, batch_norm

#### argument parser

parser = argparse.ArgumentParser(description='dfm parser')
parser.add_argument('-b', dest='batch_size', type=int, default=256)
parser.add_argument('-k', dest='k', type=int, default=32)

parser.add_argument('-init_stddev', dest='init_stddev', type=float, default=0.01)

parser.add_argument('-opt', dest='opt', type=str, default="adam")
parser.add_argument('-lr_decay', dest='lr_decay', default=False, action="store_true", help="")
parser.add_argument('-eta', dest='eta', type=float, default=0.001)

parser.add_argument('-lambda_deep', dest='lambda_deep', type=float, default=0.01)
parser.add_argument('-lambda_wide', dest='lambda_wide', type=float, default=0.01)
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

parser.add_argument('-not_use_wide', dest='not_use_wide', default=False, action="store_true")
parser.add_argument('-not_use_deep', dest='not_use_deep', default=False, action="store_true")


####


def get_test_args(parser):
    cmd = "-id all_014 -b 4 -k 32 -init_stddev 0.01 -lambda_deep 0.01 -lambda_wide 0.01 -lambda_emb 0.01 -eta 0.0001 -feat_info_path /root/feat_info_meta_20190816.json -train_path /data01/encoded_feat_20190816_train.tfrecord -test_path /data01/encoded_feat_20190816_test.tfrecord -max_epoch 200000 -date 20190817"
    args = cmd.split(" ")
    return parser.parse_args(args)


####


DW_SUMMARY_DIR = "/data01/tensorflow/summaries/dw"
DW_MODELS_DIR = "/data01/tensorflow/models/dw"


def get_context_features(n_dense):
    print("====> get_context_features, n_dense: {0}".format(n_dense))
    return {'device_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
            'req_id': tf.FixedLenFeature(shape=[1], dtype=tf.string),
            'dense': tf.FixedLenFeature(shape=[n_dense], dtype=tf.float32),
            'label': tf.FixedLenFeature(shape=[1], dtype=tf.float32)}


sequence_features = {'seq_idx': tf.VarLenFeature(dtype=tf.int64),
                     'seq_val': tf.VarLenFeature(dtype=tf.float32)}


def get_parse_seq_example_fn(n_dense):
    def parse_seq_example(example_proto):
        ex = tf.parse_single_sequence_example(example_proto, get_context_features(n_dense), sequence_features)
        return ex

    return parse_seq_example


def get_dataset_iterator(batch_size, path, n_dense):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(get_parse_seq_example_fn(n_dense))
    dataset = dataset.shuffle(buffer_size=1000000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()


#### parse feat info json

FeatInfo = collections.namedtuple('FeatInfo', 'max_feat_id dense_feat_ids slot_ids')


def parse_feat_info(path):
    with open(path) as f:
        obj = json.load(f)
    return FeatInfo(obj["max_feat_id"], obj["dense_feat_ids"], obj["slot_ids"])


####

def fc(inputs, output_size, is_training, w_initializer, activation_fn, keep_prob, use_batch_norm, bn_decay, l2_scale_deep, scope, reuse = False):
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
                                  weights_regularizer=tf.contrib.layers.l2_regularizer(l2_scale_deep),
                                  biases_regularizer=tf.contrib.layers.l2_regularizer(l2_scale_deep),
                                  scope=scope, reuse=reuse, **batch_norm_args)
        return tf.cond(is_training, lambda: tf.nn.dropout(outputs, keep_prob=keep_prob), lambda: outputs)


def pow_wrapper(X, p):
    with tf.name_scope('pow_wrapper'):
        return tf.SparseTensor(X.indices, tf.pow(X.values, p), X.dense_shape)


def check_numerics(tensor, msg, name):
    return tf.verify_tensor_all_finite(tensor, msg=msg, name=name)


def indicate_nonzero_wrapper(X):
    """
    Parameters
    ----------
    X : sparse tensor (n_samples, n_feats)
    -------
    tf.Tensor (1, n_feats)
    """
    with tf.name_scope('indicate_nonzero_wrapper'):
        indicator_X = tf.SparseTensor(X.indices, tf.ones_like(X.values), X.dense_shape)
        reduced = tf.sparse_reduce_sum(indicator_X, axis=0, keep_dims=True)
        return tf.squeeze(tf.to_float(reduced > 0))


def freeze_graph_def(sess, input_graph_def, output_node_names):
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']

    return tf.graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
