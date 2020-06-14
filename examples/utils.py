#!/usr/bin/env python
#coding=utf-8
"""
Author:deepdeliver
"""

import tensorflow as tf
from datetime import date, timedelta
from sys import argv
import argparse

parser = argparse.ArgumentParser()

def init_cmd_args(args):
    # Initialize cmd arguments
    parser.add_argument("--dist_mode", type=int, default=0, help="distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
    parser.add_argument("--ps_hosts", type=str, default='', help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--worker_hosts", type=str, default='', help="Comma-separated list of hostname:port pairs")
    parser.add_argument("--job_name", type=str, default='', help="One of 'ps', 'worker'")
    parser.add_argument("--task_index", type=int, default=0, help="Index of task within the job")
    parser.add_argument("--num_threads", type=int, default=16, help="Number of threads")
    parser.add_argument("--feature_size", type=int, default=0, help="Number of features")
    parser.add_argument("--field_size", type=int, default=0, help="Number of fields")
    parser.add_argument("--embedding_size", type=int, default=32, help="Embedding size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of batch size")
    parser.add_argument("--log_steps", type=int, default=1000, help="save summary every steps")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="learning rate")
    parser.add_argument("--l2_reg", type=float, default=0.0001, help="L2 regularization")
    parser.add_argument("--loss_type", type=str, default='log_loss', help="loss type {square_loss, log_loss}")
    parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer type {Adam, Adagrad, GD, Momentum}")
    parser.add_argument("--deep_layers", type=str, default='256,128,64', help="deep layers")
    parser.add_argument("--dropout", type=str, default='0.5,0.5,0.5', help="dropout rate")
    parser.add_argument("--batch_norm", type=bool, default=False, help="perform batch normaization (True or False)")
    parser.add_argument("--batch_norm_decay", type=float, default=0.9, help="decay for the moving average(recommend trying decay=0.9)")
    parser.add_argument("--data_dir", type=str, default='', help="data dir")
    parser.add_argument("--dt_dir", type=str, default='', help="data dt partition")
    parser.add_argument("--model_dir", type=str, default='', help="model check point dir")
    parser.add_argument("--servable_model_dir", type=str, default='', help="export servable model for TensorFlow Serving")
    parser.add_argument("--task_type", type=str, default='train', help="task type {train, infer, eval, export}")
    parser.add_argument("--clear_existing_model", type=bool, default=False, help="clear existing model or not")
    args_parser = parser.parse_args(args[1:])

    # check arguments
    if args_parser.dt_dir == "":
        args_parser.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    args_parser.model_dir = args_parser.model_dir + args_parser.dt_dir

    print('task_type ', args_parser.task_type)
    print('model_dir ', args_parser.model_dir)
    print('data_dir ', args_parser.data_dir)
    print('dt_dir ', args_parser.dt_dir)
    print('num_epochs ', args_parser.num_epochs)
    print('feature_size ', args_parser.feature_size)
    print('field_size ', args_parser.field_size)
    print('embedding_size ', args_parser.embedding_size)
    print('batch_size ', args_parser.batch_size)
    print('deep_layers ', args_parser.deep_layers)
    print('dropout ', args_parser.dropout)
    print('loss_type ', args_parser.loss_type)
    print('optimizer ', args_parser.optimizer)
    print('learning_rate ', args_parser.learning_rate)
    print('batch_norm_decay ', args_parser.batch_norm_decay)
    print('batch_norm ', args_parser.batch_norm)
    print('l2_reg ', args_parser.l2_reg)
    return args_parser

def set_dist_env(args_parser):
    if args_parser.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = args_parser.ps_hosts.split(',')
        chief_hosts = args_parser.chief_hosts.split(',')
        task_index = args_parser.task_index
        job_name = args_parser.job_name
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
    elif args_parser.dist_mode == 2:      # 集群分布式模式
        ps_hosts = args_parser.ps_hosts.split(',')
        worker_hosts = args_parser.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = args_parser.task_index
        job_name = args_parser.job_name
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
