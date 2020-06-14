#!/usr/bin/env python
#coding=utf-8
"""
Author:deepdeliver
"""

import tensorflow as tf
from datetime import date, timedelta

FLAGS = tf.app.flags.FLAGS

def init_cmd_args():
    # Initialize cmd arguments
    tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
    tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
    tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
    tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
    tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
    tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
    tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
    tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
    tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
    tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
    tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
    tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
    tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
    tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
    tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
    tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
    tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
    tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
    tf.app.flags.DEFINE_string("data_dir", '', "data dir")
    tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
    tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
    tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
    tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
    tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

    # check arguments
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

def set_dist_env():
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
