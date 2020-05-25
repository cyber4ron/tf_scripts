# -*- encoding: utf-8 -*-

import time

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer

import yt_dnn_utils as utils

####

YT_DNN_SUMMARY_DIR = "/data01/tensorflow/summaries/yt_dnn"
YT_DNN_MODELS_DIR = "/data01/tensorflow/models/yt_dnn"

args = utils.parser.parse_args()
print("====> args:", args)

feat_info = utils.parse_feat_info(args.feat_info_path)
print("====> feat info:", feat_info)

####

task_id = args.id

n_feature = feat_info.max_feat_id + 1  # 包括n_dense, 不过简单起见，声明n_feature个embeddings
n_dense = len(feat_info.dense_feat_ids)
n_field = len(feat_info.slot_ids)

pub_id_offset = feat_info.pub_id_offset
pub_id_num = feat_info.pub_id_num

k = args.k
init_stddev = args.init_stddev

use_batch_norm = True
bn_decay = 0.999
l2_scale_nn = args.lambda_nn
l2_scale_emb = args.lambda_emb

hidden_layers = [512, 256, k]
n_hidden_layers = len(hidden_layers)
dropouts = [0.7, 0.7, 0.7]

eta = args.eta
batch_size = args.batch_size

custom_loss = args.custom_loss

epoch_max = args.max_epoch

pub_id_mapping_path = args.pub_id_mapping_path

train_path = args.train_path
test_path = args.test_path

date = args.date
for_eval = args.for_eval

iterator = utils.get_dataset_iterator(batch_size, train_path, n_dense)
test_iterator = utils.get_dataset_iterator(batch_size, test_path, n_dense)

####

with tf.name_scope('custom_variables'):
    emb = tf.Variable(tf.truncated_normal([n_feature, k], stddev=init_stddev), name="emb")  # range: [0, n_feature)
    emb_zeros = tf.Variable(tf.zeros([n_field + 1, k]), trainable=False,
                            name="emb_zeros")  # range: [n_feature, n_feature + n_field + 1 (for y_id padding))
    emb_all = tf.concat([emb, emb_zeros], axis=0, name="emb_all")
    emb_pub_id = tf.slice(emb_all, [pub_id_offset, 0], [pub_id_num, k], "emb_pub_id")

with tf.name_scope('inputs'):
    is_training = tf.placeholder_with_default(False, shape=[], name="is_training")
    dense = tf.placeholder(tf.float32, shape=[None, n_dense], name="dense")
    #
    seq_idx_indices = tf.placeholder(tf.int64, shape=[None, 3], name="seq_idx_indices")
    seq_idx_values = tf.placeholder(tf.int64, shape=[None], name="seq_idx_values")
    seq_idx_shape = tf.placeholder(tf.int64, shape=[3], name="seq_idx_shape")
    seq_idx = tf.SparseTensor(seq_idx_indices, seq_idx_values, seq_idx_shape)
    #
    seq_val_indices = tf.placeholder(tf.int64, shape=[None, 3], name="seq_val_indices")
    seq_val_values = tf.placeholder(tf.float32, shape=[None], name="seq_val_values")
    seq_val_shape = tf.placeholder(tf.int64, shape=[3], name="seq_val_shape")
    seq_val = tf.SparseTensor(seq_val_indices, seq_val_values, seq_val_shape)
    #
    y_pos = tf.sparse.placeholder(tf.int64, shape=[None, 1], name='y_pos')
    y_neg = tf.sparse.placeholder(tf.int64, shape=[None, 1], name='y_neg')

with tf.name_scope('embeddings'):
    sp_ids = tf.sparse.reshape(seq_idx, [-1, tf.shape(seq_idx)[-1]], name="sp_ids")
    sp_weights = tf.sparse.reshape(seq_val, [-1, tf.shape(seq_val)[-1]], name="sp_weights")
    tmp_emb = tf.nn.embedding_lookup_sparse(emb_all, sp_ids=sp_ids, sp_weights=sp_weights, combiner="mean")  # todo: 检查, 大小; sp_weights
    seq_emb = tf.reshape(tmp_emb, [-1, n_field * k], name="seq_emb")
    #
    deep_input = tf.concat([dense, seq_emb], axis=1)

with tf.name_scope('inference'):
    deep = deep_input
    for i in range(0, n_hidden_layers):
        act_fn = None if i == n_hidden_layers - 1 else tf.nn.relu
        init_fn = xavier_initializer() if i == n_hidden_layers - 1 else variance_scaling_initializer()
        deep = utils.fc(deep, hidden_layers[i], is_training, w_initializer=init_fn, activation_fn=act_fn, keep_prob=dropouts[i],
                        use_batch_norm=use_batch_norm, bn_decay=bn_decay, l2_scale_nn=l2_scale_nn, scope="layer%d" % i)
    #
    deep_out = tf.identity(deep, name="deep_out")
    #
    y_id, y_label, min_row_len = utils.get_padded_pos_neg(y_pos, y_neg, n_feature + n_field)
    weights = tf.nn.embedding_lookup(emb_all, y_id)
    #
    deep_reshaped = tf.reshape(deep_out, [batch_size, 1, k])  # [batch_size, 1, k]
    weights_transposed = tf.transpose(weights, [0, 2, 1])  # [batch_size, k, pos+neg]
    logits = tf.squeeze(tf.matmul(deep_reshaped, weights_transposed))  # [batch_size, pos+neg]
    logits_sliced = tf.slice(logits, [0, 0], [batch_size, min_row_len])

with tf.name_scope('loss'):
    y_label_sliced = tf.slice(y_label, [0, 0], [batch_size, min_row_len])
    if custom_loss:
        y_prob_sliced = tf.nn.softmax(logits_sliced)
        ce_loss = tf.reduce_mean(-tf.cast(y_label_sliced, tf.float32) * tf.log(y_prob_sliced + 1e-7))
    else:
        ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label_sliced, logits=logits_sliced))
    #
    unique_ids, _ = tf.unique(seq_idx_values, out_idx=tf.int64)
    embedding_slices = tf.nn.embedding_lookup(emb_all, unique_ids)
    emb_l2_loss = l2_scale_emb * tf.nn.l2_loss(embedding_slices, name="emb_l2_loss")
    loss = ce_loss + emb_l2_loss

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=eta)
    opt_op = optimizer.minimize(loss)

####

with tf.name_scope('metric'):
    prob_sliced = tf.nn.softmax(logits_sliced)
    prob_argmax = tf.argmax(prob_sliced, axis=1)
    prob_top_5, prob_idx_top_5 = tf.math.top_k(prob_sliced, k=5)
    #
    _, p_1 = tf.metrics.precision_at_k(y_label_sliced, prob_sliced, k=1)
    _, p_5 = tf.metrics.precision_at_k(y_label_sliced, prob_sliced, k=5)
    _, r_1 = tf.metrics.recall_at_k(y_label_sliced, prob_sliced, k=1)
    _, r_5 = tf.metrics.recall_at_k(y_label_sliced, prob_sliced, k=5)
    _, ap_5 = tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=5)
    #
    # map_5, shorter
    map_5 = (tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=1)[1] +
             tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=2)[1] +
             tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=3)[1] +
             tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=4)[1] +
             tf.metrics.average_precision_at_k(y_label_sliced, prob_sliced, k=5)[1]) / 5
    #
    tf.summary.scalar("min_row_len", min_row_len)
    #
    tf.summary.histogram("emb", emb)
    tf.summary.histogram("emb_zeros", emb_zeros)
    #
    tf.summary.scalar("ce_loss", ce_loss)
    tf.summary.scalar("emb_l2_loss", emb_l2_loss)
    tf.summary.scalar('loss', loss)
    #
    tf.summary.scalar('p_1', p_1)
    tf.summary.scalar('p_5', p_5)
    tf.summary.scalar('r_1', r_1)
    tf.summary.scalar('r_5', r_5)
    tf.summary.scalar('ap_5', ap_5)
    tf.summary.scalar('map_5', map_5)


####

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


def save_embeddings(pub_id_mappings):
    print("====> saving publisher embeddings...")
    publisher_embeddings = sess.run(emb_pub_id)
    #
    pub_id = pub_id_offset
    path = "{0}/{1}/pub_emb_id_embeddings.txt".format(YT_DNN_MODELS_DIR, task_id)
    with open(path, 'w') as f:
        for e in publisher_embeddings:
            f.write("{0}\t{1}\n".format(pub_id, ','.join([str(f) for f in e])))
            pub_id += 1
    #
    pub_id = pub_id_offset
    path = "{0}/{1}/publisher_embeddings.txt".format(YT_DNN_MODELS_DIR, task_id)
    with open(path, 'w') as f:
        for e in publisher_embeddings:
            f.write("{0}\t{1}\n".format(pub_id_mappings[pub_id - pub_id_offset], ','.join([str(f) for f in e])))
            pub_id += 1


####

metrics_var = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
metrics_var_init = tf.variables_initializer(var_list=metrics_var)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('{0}/{1}/train'.format(YT_DNN_SUMMARY_DIR, task_id), sess.graph)
    test_writer = tf.summary.FileWriter('{0}/{1}/test'.format(YT_DNN_SUMMARY_DIR, task_id), sess.graph)
    sess.run([tf.global_variables_initializer(), iterator.initializer, test_iterator.initializer])
    element = iterator.get_next()
    test_element = test_iterator.get_next()
    #
    for epoch in range(int(1e8)):
        sess.run(metrics_var_init)
        start = time.time()
        x = sess.run(element)
        fetches = [deep_out, min_row_len, prob_argmax, y_label_sliced, logits_sliced, prob_idx_top_5, loss, opt_op, merged]
        d_o_, m_r_l_, p_a_m_, y_l_s_, l_s_, p_t5_, loss_, _, summary = sess.run(fetches, feed_dict={is_training: True,
                                                                                                    dense: x[0]['dense'],
                                                                                                    seq_idx: x[1]['seq_idx'],
                                                                                                    seq_val: x[1]['seq_val'],
                                                                                                    y_pos: x[0]['pos'],
                                                                                                    y_neg: x[0]['neg']})
        #
        print("====> epoch {0}, min_row_len: {1}, prob_argmax: {2}, prob_idx_top_5: {7}, y_label_sliced: {3}, logits_sliced: {4}, "
              "neg shape: {5}, seq_idx shape: {6}, deep_out: {8}".format(epoch, m_r_l_, p_a_m_, y_l_s_, l_s_, x[0]['neg'].dense_shape,
                                                                         x[1]['seq_idx'].dense_shape, p_t5_, d_o_))
        writer.add_summary(summary, epoch)
        writer.flush()
        print('train, epoch: {0}, loss: {1}, elapsed:{2}, '.format(epoch, loss_, time.time() - start))
        if epoch % 10 == 0:
            sess.run(metrics_var_init)
            start = time.time()
            x = sess.run(test_element)
            fetches = [deep_out, prob_argmax, y_label_sliced, logits_sliced, loss, merged]
            d_o_, p_a_m_, y_l_s_, l_s_, loss_, summary = sess.run(fetches, feed_dict={is_training: True,
                                                                                      dense: x[0]['dense'],
                                                                                      seq_idx: x[1]['seq_idx'],
                                                                                      seq_val: x[1]['seq_val'],
                                                                                      y_pos: x[0]['pos'],
                                                                                      y_neg: x[0]['neg']})
            test_writer.add_summary(summary, epoch)
            test_writer.flush()
            print('test, epoch: {0}, loss: {1}, elapsed: {2}, seq_idx shape: {3}'.format(epoch, loss_, time.time() - start,
                                                                                         x[1]['seq_idx'].dense_shape))
        #
        if epoch >= epoch_max:
            print("====> all nodes:")
            for n in sess.graph.as_graph_def().node:
                print(n.name)
            #
            # freeze graph
            graph_def = sess.graph.as_graph_def()
            output_graph_def = freeze_graph_def(sess, graph_def, "inference/deep_out")
            #
            # import graph_def and create new session
            graph_export = tf.Graph()
            with graph_export.as_default():
                tf.import_graph_def(output_graph_def, name="")
            sess_export = tf.Session(graph=graph_export)
            #
            print("====> all nodes, after freeze:")
            for n in sess_export.graph.as_graph_def().node:
                print(n.name)
            #
            print('====> saving model...')
            export_dir = "{0}/{1}/{2}".format(YT_DNN_MODELS_DIR, task_id, date)
            with sess_export.graph.as_default():
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                builder.add_meta_graph_and_variables(
                    sess_export, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={
                                "is_training": tf.saved_model.utils.build_tensor_info(is_training),
                                "dense": tf.saved_model.utils.build_tensor_info(dense),
                                #
                                "seq_idx_indices": tf.saved_model.utils.build_tensor_info(seq_idx_indices),
                                "seq_idx_values": tf.saved_model.utils.build_tensor_info(seq_idx_values),
                                "seq_idx_shape": tf.saved_model.utils.build_tensor_info(seq_idx_shape),
                                #
                                "seq_val_values": tf.saved_model.utils.build_tensor_info(seq_val_values)
                            },
                            outputs={
                                'user_embedding': tf.saved_model.utils.build_tensor_info(deep_out),
                            },
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                    })
            builder.save()
            print('====> model saved.')
            #
            # save embeddings
            save_embeddings(utils.load_pub_id_mapping(pub_id_mapping_path))
            #
            # test save model
            with tf.Session(graph=tf.Graph()) as test_sess:
                meta = tf.saved_model.loader.load(test_sess, [tf.saved_model.tag_constants.SERVING], export_dir)
                inputs_mapping = dict(meta.signature_def['serving_default'].inputs)
                outputs_mapping = dict(meta.signature_def['serving_default'].outputs)
                print("====> inputs_mapping: {0}".format(inputs_mapping))
                print("====> outputs_mapping: {0}".format(outputs_mapping))
                #
                graph = tf.get_default_graph()
                print("====> all nodes in imported graph:")
                for n in graph.as_graph_def().node:
                    print(n.name)
                #
                is_training = graph.get_tensor_by_name("inputs/is_training:0")
                dense = graph.get_tensor_by_name("inputs/dense:0")
                #
                seq_idx_indices = graph.get_tensor_by_name("inputs/seq_idx_indices:0")
                seq_idx_values = graph.get_tensor_by_name("inputs/seq_idx_values:0")
                seq_idx_shape = graph.get_tensor_by_name("inputs/seq_idx_shape:0")
                #
                seq_val_values = graph.get_tensor_by_name("inputs/seq_val_values:0")
                #
                embedding = graph.get_tensor_by_name("inference/deep_out:0")
                print(test_sess.run(embedding, feed_dict={is_training: False,
                                                     dense: x[0]['dense'],
                                                     #
                                                     seq_idx_indices: x[1]['seq_idx'].indices,
                                                     seq_idx_values: x[1]['seq_idx'].values,
                                                     seq_idx_shape: x[1]['seq_idx'].dense_shape,
                                                     #
                                                     seq_val_values: x[1]['seq_val'].values
                                                     }))
            #
            break
    writer.close()
    test_writer.close()
