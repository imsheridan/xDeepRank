#!/usr/bin/env python
#coding=utf-8
"""
Author:deepdeliver
"""
import glob
import utils
import random
import shutil
import tensorflow as tf
import sys
sys.path.append("../")

from xdeeprank.models import DeepFM
from xdeeprank.inputs import input_fn

FLAGS = tf.app.flags.FLAGS

def main(_):
    # init environments
    utils.init_cmd_args()

    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    utils.set_dist_env()

    # build model params
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm": FLAGS.batch_norm,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "optimizer": FLAGS.optimizer
    }

    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    model = tf.estimator.Estimator(model_fn=DeepFM, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
