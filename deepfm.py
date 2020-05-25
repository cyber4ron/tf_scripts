# coding=utf-8

from deepfm_utils import parser

import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorflow.contrib.layers import variance_scaling_initializer, xavier_initializer

import deepfm_utils as utils

args = parser.parse_args()
print("====> args:", args)

feat_info = utils.parse_feat_info(args.feat_info_path)
print("====> feat info:", feat_info)

####

task_id = args.id

n_feature = feat_info.max_feat_id + 1  # 包括n_dense, 不过简单起见，声明n_feature个embeddings
n_dense = len(feat_info.dense_feat_ids)
n_field = len(feat_info.slot_ids)

hidden_layers = [128, 128, 1]
n_hidden_layers = len(hidden_layers)
dropouts = [0.8, 0.8, 1]

k = args.k
init_stddev = args.init_stddev

use_batch_norm = True
bn_decay = 0.999
l2_scale_deep = args.lambda_deep
l2_scale_wide = args.lambda_wide
l2_scale_emb = args.lambda_emb

opt = args.opt
eta = args.eta

batch_size = args.batch_size

train_path = args.train_path
test_path = args.test_path

iterator = utils.get_dataset_iterator(batch_size, train_path, n_dense)
test_iterator = utils.get_dataset_iterator(batch_size, test_path, n_dense)

not_use_wide = args.not_use_wide
not_use_deep = args.not_use_deep

epoch_max = args.max_epoch
date = args.date

####

with tf.name_scope('custom_variables'):
    # first order
    w0 = tf.Variable(tf.zeros([1]), name="w0")
    W = tf.Variable(tf.truncated_normal([n_feature, 1], stddev=init_stddev), name="W")
    W_zero = tf.Variable(tf.truncated_normal([n_field, 1], stddev=init_stddev), trainable=False, name="W_zero")
    W_all = tf.concat([W, W_zero], axis=0, name="W_all")
    #
    # 2 order & deep
    emb = tf.Variable(tf.truncated_normal([n_feature, k], stddev=init_stddev), name="emb")  # range: [0, n_feature)
    emb_zeros = tf.Variable(tf.zeros([n_field, k]), trainable=False, name="emb_zeros")
    emb_all = tf.concat([emb, emb_zeros], axis=0, name="emb_all")  # range: [n_feature, n_feature + n_field)

with tf.name_scope('inputs'):
    is_training = tf.placeholder_with_default(False, shape=[], name="is_training")
    b_size = tf.placeholder(tf.int64, name="batch_size")
    dense = tf.placeholder(tf.float32, shape=[None, n_dense], name="dense")  # (N, n_dense)
    #
    seq_idx_indices = tf.placeholder(tf.int64, shape=[None, 3], name="seq_idx_indices")
    seq_idx_values = tf.placeholder(tf.int64, shape=[None], name="seq_idx_values")  # feat id
    seq_idx_shape = tf.placeholder(tf.int64, shape=[3], name="seq_idx_shape")
    seq_idx = tf.SparseTensor(seq_idx_indices, seq_idx_values, seq_idx_shape)
    #
    seq_val_indices = tf.placeholder(tf.int64, shape=[None, 3], name="seq_val_indices")
    seq_val_values = tf.placeholder(tf.float32, shape=[None], name="seq_val_values")  # feat value
    seq_val_shape = tf.placeholder(tf.int64, shape=[3], name="seq_val_shape")
    seq_val = tf.SparseTensor(seq_val_indices, seq_val_values, seq_val_shape)
    #
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

with tf.name_scope('embeddings'):
    # for deep
    sp_ids = tf.sparse.reshape(seq_idx, [-1, tf.shape(seq_idx)[-1]], name="sp_ids")  # [N * n_field, max_seq_len_in_batch]
    sp_weights = tf.sparse.reshape(seq_val, [-1, tf.shape(seq_val)[-1]], name="sp_weights")  # [N * n_field, max_seq_len_in_batch]
    tmp_emb = tf.nn.embedding_lookup_sparse(emb_all, sp_ids=sp_ids, sp_weights=sp_weights, combiner="mean")
    seq_emb = tf.reshape(tmp_emb, [-1, n_field * k], name="seq_emb")
    #
    deep_input = tf.concat([dense, seq_emb], axis=1)

with tf.name_scope('inference'):
    # wide, todo: test
    spr_ids = tf.sparse.reshape(seq_idx, [b_size, -1])  # (batch_size, max_n_feat_in_batch), 只有sparse特征，没有dense
    print("====> spr_ids shape:", spr_ids.get_shape())
    #
    # first-order
    linear = tf.add(w0, tf.nn.embedding_lookup_sparse(W_all, spr_ids, None, combiner="sum"))  # (batch_size, 1(reduce_sum掉了))
    print("====> linear shape:", linear.get_shape())
    #
    # second-order
    interaction = tf.multiply(0.5, tf.reduce_sum(
        tf.subtract(
            tf.square(tf.nn.embedding_lookup_sparse(emb_all, spr_ids, None, combiner="sum")),
            tf.nn.embedding_lookup_sparse(tf.square(emb_all), spr_ids, None, combiner="sum")),
        1,
        keep_dims=True))  # (batch_size, 1(reduce_sum掉了)),
    print("====> interaction shape:", interaction.get_shape())
    #
    # deep
    deep = deep_input
    for i in range(0, n_hidden_layers):
        act_fn = None if i == n_hidden_layers - 1 else tf.nn.relu
        init_fn = xavier_initializer() if i == n_hidden_layers - 1 else variance_scaling_initializer()
        deep = utils.fc(deep, hidden_layers[i], is_training, w_initializer=init_fn, activation_fn=act_fn, keep_prob=dropouts[i],
                        use_batch_norm=use_batch_norm, bn_decay=bn_decay, l2_scale_deep=l2_scale_deep, scope="layer%d" % i)
    #
    # deep - (batch_size, 1)
    print("====> deep shape:", deep.get_shape())
    if not_use_wide:
        logits = deep
    elif not_use_deep:
        logits = linear + interaction
    else:
        logits = linear + interaction + deep
    print("====> logits shape:", logits.get_shape())  # (batch_size, 1)
    # print("====> linear.get_shape(): {0}, interaction.get_shape(): {1}, deep.get_shape(): {2}".format(linear.get_shape(),
    #                                                                                                   interaction.get_shape(),
    #                                                                                                   deep.get_shape()))
    y_hat = tf.sigmoid(logits, name="y_hat")
    print("====> y_hat shape:", y_hat.get_shape())  # (batch_size, 1)

with tf.name_scope('loss'):
    #
    wide_log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=linear + interaction))
    deep_log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=deep))
    log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits))
    #
    nz_idx, _ = tf.unique(sp_ids.values)
    indicators = tf.scatter_nd(tf.expand_dims(nz_idx, -1),
                               tf.fill(tf.shape(nz_idx), tf.constant(1.0, dtype=tf.float32)),
                               shape=[n_feature + n_field])
    #
    wide_l2_loss = l2_scale_wide * tf.add(
        tf.reduce_sum(tf.pow(tf.multiply(tf.reshape(W_all, shape=[n_feature + n_field]), indicators, name="W_masked"), 2)),
        tf.reduce_sum(tf.pow(tf.multiply(tf.transpose(emb_all), indicators, name="emb_masked"), 2)))
    #
    deep_l2_loss = tf.losses.get_regularization_loss()
    print("====> log_loss.get_shape: {0}, custom_l2_loss.get_shape: {1}, internal_l2_loss.get_shape: {2}"
          .format(log_loss.get_shape(), wide_l2_loss.get_shape(), deep_l2_loss.get_shape()))
    #
    # unique_ids, _ = tf.unique(seq_idx_values, out_idx=tf.int64)
    # embedding_slices = tf.nn.embedding_lookup(emb_all, unique_ids)
    # emb_l2_loss = l2_scale_emb * tf.nn.l2_loss(embedding_slices, name="emb_l2_loss")
    #
    # loss = log_loss + wide_l2_loss + deep_l2_loss
    loss = log_loss

with tf.name_scope('train'):
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(eta, global_step, 10000, 0.9, staircase=False)
    #
    if opt == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tf.summary.scalar('optimizer_lr', optimizer._lr)
        print("====> using adam:", opt)
    elif opt == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        tf.summary.scalar('optimizer_lr', optimizer._learning_rate)
        print("====> using adagrad:", opt)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        tf.summary.scalar('optimizer_lr', optimizer._learning_rate)
        print("====> using gd:", opt)
    #
    opt_op = optimizer.minimize(loss, global_step=global_step)

with tf.name_scope('metric'):
    tf.summary.histogram("bias", w0)
    tf.summary.histogram("W", W)
    tf.summary.histogram("W_all", W_all)
    tf.summary.histogram("V", emb_all)
    tf.summary.histogram("V0", emb_zeros)
    #
    with tf.variable_scope("", reuse=True):
        tf.summary.histogram("layer0/weights", tf.get_variable("layer0/layer0/weights"))
        tf.summary.histogram("layer0/bn/beta", tf.get_variable("layer0/layer0/BatchNorm/beta"))
        tf.summary.histogram("layer0/bn/gamma", tf.get_variable("layer0/layer0/BatchNorm/gamma"))
        tf.summary.histogram("layer2/weights", tf.get_variable("layer2/layer2/weights"))
        tf.summary.histogram("layer2/bn/beta", tf.get_variable("layer2/layer2/BatchNorm/beta"))
        tf.summary.histogram("layer2/bn/gamma", tf.get_variable("layer2/layer2/BatchNorm/gamma"))
    #
    for g in optimizer.compute_gradients(loss):
        if g[0] is not None:
            tf.summary.histogram("%s-grad" % g[1].name, g[0])
    #
    y_hat_round = tf.round(y_hat)
    _, acc_op = tf.metrics.accuracy(labels=y, predictions=y_hat_round)
    _, prec_op = tf.metrics.precision(labels=y, predictions=y_hat_round)
    _, rec_op = tf.metrics.recall(labels=y, predictions=y_hat_round)
    _, auc_op = tf.metrics.auc(labels=y, predictions=y_hat)
    #
    tf.summary.scalar('accuracy', acc_op)
    tf.summary.scalar('precision', prec_op)
    tf.summary.scalar('recall', rec_op)
    tf.summary.scalar('auc', auc_op)
    #
    tf.summary.scalar("wide_log_loss", wide_log_loss)
    tf.summary.scalar("deep_log_loss", deep_log_loss)
    tf.summary.scalar("log_loss", log_loss)
    tf.summary.scalar('custom_l2_loss', wide_l2_loss)
    # tf.summary.scalar('emb_l2_loss', emb_l2_loss)
    tf.summary.scalar('internal_l2_loss', deep_l2_loss)
    tf.summary.scalar('loss', loss)

N_EPOCHS = 150000

metrics_var = tf.get_collection(tf.GraphKeys.METRIC_VARIABLES)
metrics_var_init = tf.variables_initializer(var_list=metrics_var)
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('{0}/{1}/train'.format(utils.DW_SUMMARY_DIR, task_id), sess.graph)
    test_writer = tf.summary.FileWriter('{0}/{1}/test'.format(utils.DW_SUMMARY_DIR, task_id), sess.graph)
    sess.run([tf.global_variables_initializer(), iterator.initializer, test_iterator.initializer])
    element = iterator.get_next()
    test_element = test_iterator.get_next()
    #
    for epoch in range(N_EPOCHS):
        sess.run(metrics_var_init)
        start = time.time()
        x = sess.run(element)  # , merged
        loss_, _, summary = sess.run([loss, opt_op, merged], feed_dict={is_training: True,
                                                                        b_size: batch_size,
                                                                        dense: x[0]['dense'],
                                                                        seq_idx: x[1]['seq_idx'],
                                                                        seq_val: x[1]['seq_val'],
                                                                        y: x[0]['label']})
        writer.add_summary(summary, epoch)
        writer.flush()
        print('train, epoch: {0}, loss: {1}, elapsed:{2}'.format(epoch, loss_, time.time() - start))
        if epoch % 20 == 0:
            sess.run(metrics_var_init)
            start = time.time()
            x = sess.run(test_element)
            y_, y_hat_, loss_, summary = sess.run([y, y_hat, loss, merged], feed_dict={is_training: False,
                                                                                       b_size: batch_size,
                                                                                       dense: x[0]['dense'],
                                                                                       seq_idx: x[1]['seq_idx'],
                                                                                       seq_val: x[1]['seq_val'],
                                                                                       y: x[0]['label']})
            fpr, tpr, thresholds = roc_curve(np.squeeze(y_), np.squeeze(y_hat_))  # thresholds: Decreasing thresholds
            print("====> y_: {0}, y_hat_: {1}, fpr: {2}, tpr: {3}".format(np.squeeze(y_), np.squeeze(y_hat_), fpr, tpr))
            au_roc = auc(fpr, tpr)
            print('test, epoch: {0}, sklearn, au_roc: {1}'.format(epoch, au_roc))
            test_writer.add_summary(summary, epoch)
            test_writer.flush()
            print('test, epoch: {0}, loss: {1}, elapsed: {2}'.format(epoch, loss_, time.time() - start))
        if epoch >= epoch_max:
            print("====> all nodes:")
            for n in sess.graph.as_graph_def().node:
                print(n.name)
            #
            # freeze graph
            graph_def = sess.graph.as_graph_def()
            output_graph_def = utils.freeze_graph_def(sess, graph_def, "inference/y_hat")
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
            export_dir = "{0}/{1}/{2}".format(utils.DW_MODELS_DIR, task_id, date)
            with sess_export.graph.as_default():
                builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
                builder.add_meta_graph_and_variables(
                    sess_export, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={
                                "is_training": tf.saved_model.utils.build_tensor_info(is_training),
                                "batch_size": tf.saved_model.utils.build_tensor_info(b_size),
                                "dense": tf.saved_model.utils.build_tensor_info(dense),
                                #
                                "seq_idx_indices": tf.saved_model.utils.build_tensor_info(seq_idx_indices),
                                "seq_idx_values": tf.saved_model.utils.build_tensor_info(seq_idx_values),
                                "seq_idx_shape": tf.saved_model.utils.build_tensor_info(seq_idx_shape),
                                #
                                "seq_val_values": tf.saved_model.utils.build_tensor_info(seq_val_values)
                            },
                            outputs={
                                'y_hat': tf.saved_model.utils.build_tensor_info(y_hat),
                            },
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
                    })
            builder.save()
            print('====> model saved.')
            #
            # test save model, 抽成方法
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
                b_size = graph.get_tensor_by_name("inputs/batch_size:0")
                #
                seq_idx_indices = graph.get_tensor_by_name("inputs/seq_idx_indices:0")
                seq_idx_values = graph.get_tensor_by_name("inputs/seq_idx_values:0")
                seq_idx_shape = graph.get_tensor_by_name("inputs/seq_idx_shape:0")
                #
                seq_val_values = graph.get_tensor_by_name("inputs/seq_val_values:0")
                #
                embedding = graph.get_tensor_by_name("inference/y_hat:0")
                print(test_sess.run(embedding, feed_dict={is_training: False,
                                                          dense: x[0]['dense'],
                                                          b_size: batch_size,
                                                          #
                                                          seq_idx_indices: x[1]['seq_idx'].indices,
                                                          seq_idx_values: x[1]['seq_idx'].values,
                                                          seq_idx_shape: x[1]['seq_idx'].dense_shape,
                                                          #
                                                          seq_val_values: x[1]['seq_val'].values
                                                          }))
            #
            break
    #
    writer.close()
    test_writer.close()
