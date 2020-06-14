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
from sys import argv

def main(_):
    # init environments
    args_parser = utils.init_cmd_args(argv)

    tr_files = glob.glob("%s/tr*libsvm" % args_parser.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % args_parser.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % args_parser.data_dir)
    print("te_files:", te_files)

    if args_parser.clear_existing_model:
        try:
            shutil.rmtree(args_parser.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % args_parser.model_dir)

    utils.set_dist_env(args_parser)

    # build model params
    model_params = {
        "field_size": args_parser.field_size,
        "feature_size": args_parser.feature_size,
        "embedding_size": args_parser.embedding_size,
        "learning_rate": args_parser.learning_rate,
        "batch_norm": args_parser.batch_norm,
        "batch_norm_decay": args_parser.batch_norm_decay,
        "l2_reg": args_parser.l2_reg,
        "deep_layers": args_parser.deep_layers,
        "dropout": args_parser.dropout,
        "optimizer": args_parser.optimizer
    }

    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':args_parser.num_threads}),
            log_step_count_steps=args_parser.log_steps, save_summary_steps=args_parser.log_steps)
    model = tf.estimator.Estimator(model_fn=DeepFM, model_dir=args_parser.model_dir, params=model_params, config=config)

    if args_parser.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=args_parser.num_epochs, batch_size=args_parser.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=args_parser.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    elif args_parser.task_type == 'eval':
        model.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=args_parser.batch_size))
    elif args_parser.task_type == 'infer':
        preds = model.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=args_parser.batch_size), predict_keys="prob")
        with open(args_parser.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif args_parser.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, args_parser.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, args_parser.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        model.export_savedmodel(args_parser.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
