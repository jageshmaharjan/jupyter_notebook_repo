from __future__ import print_function

import argparse
import logging
import os
import random

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.contrib.data import TFRecordDataset

from deepspeaker.utils.generic_utils import run_once, threadsafe_generator, initialize_logger


class SampleDataGenerator(object):
    """
    Sample DataGenerator
    1. TfRecords for handling very big data sets which cannot fit in memory
    2. Tensorflow Dataset api for creating complex pipelines (like data augumentation techniques)
    3. In built efficient data generators through cache, multi-threading and queues in Dataset API
    """

    def __init__(self, train_filenames, validation_filenames,
                 test_filenames, batch_size=64, model_type='dnn', num_threads=4):
        """
        :param train_filenames: Train TfRecord files
        :param validation_filenames: validation TfRecord files
        :param test_filenames: Test TfRecord files
        :param batch_size: Batch size
        :param model_type: Type of model consuming the data.
            If it is dnn 2D tensor is returned (batch_size, time * freq)
            If it is rnn 3D tensor is returned (batch_size, time, freq)
            If it is cnn 4D tensor is returned (batch_size, time, freq, 1)
        :param num_threads: Number of threads for enqueueing data
        """
        # For reproducibility
        random.seed(42)
        np.random.seed(42)

        self.batch_size = batch_size
        self.train_filenames = train_filenames
        self.validation_filenames = validation_filenames
        self.test_filenames = test_filenames
        self.model_type = model_type
        self.num_threads = num_threads

        self.shape, self.num_labels, self.train_spe, self.validation_spe, self.test_spe = self.get_data_info()

        self.train_x_batch, self.train_y_batch = self.prepare_dataset(self.train_filenames)
        self.validation_x_batch, self.validation_y_batch = self.prepare_dataset(self.validation_filenames)
        self.test_x_batch, self.test_y_batch = self.prepare_dataset(self.test_filenames)

        self.coordinator = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coordinator, sess=K.get_session())

    @staticmethod
    def count_samples(filename):
        """
        Count number of samples in TfRecord file
        :param filename: TfRecord file path
        :return:
        """
        return np.sum([1 for _ in tf.python_io.tf_record_iterator(filename)])

    @run_once
    def get_data_info(self):
        """
        Returns shape of data, number of labels, steps per epoch of training, validation and test
        """
        dataset = TFRecordDataset(self.train_filenames)
        dataset = dataset.map(self.parser)
        dataset = dataset.take(4)
        iterator = dataset.make_one_shot_iterator()
        sample_data = K.get_session().run(
            iterator.get_next()
        )

        train_spe = int(np.ceil(self.count_samples(self.train_filenames) * 1.0 / self.batch_size))
        validation_spe = int(np.ceil(self.count_samples(self.validation_filenames) * 1.0 / self.batch_size))
        test_spe = int(np.ceil(self.count_samples(self.test_filenames) * 1.0 / self.batch_size))

        logging.info("Shape of input data: {}".format(sample_data[0].shape))
        logging.info("Number of labels in input data: {}".format(sample_data[1].size))
        logging.info("Steps per epoch - Train: {}, Validation: {}, Test: {}".format(
            train_spe, validation_spe, test_spe))

        return sample_data[0].shape, sample_data[1].size, train_spe, validation_spe, test_spe

    def parser(self, record):
        """
        Parse a TfRecord
        :param record: A TfRecord
        :return: x node: Feature, y node: Label
        """
        features = tf.parse_single_example(record, features={
            'x': tf.FixedLenFeature([], tf.string),
            'y': tf.FixedLenFeature([], tf.string),
            't_dim': tf.FixedLenFeature([], tf.int64),
            'f_dim': tf.FixedLenFeature([], tf.int64),
            'file_id': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
        })
        x = tf.decode_raw(features['x'], tf.float32)
        y = tf.decode_raw(features['y'], tf.float32)
        t_dim = tf.cast(features['t_dim'], tf.int32)
        f_dim = tf.cast(features['f_dim'], tf.int32)
        x = self.transform_input(K.reshape(x, K.stack([t_dim, f_dim])))
        return x, y

    def prepare_dataset(self, filename):
        """
        Datset transformation pipeline
        :param filename: TfRecord filename
        :return: iterator of x, y batch nodes where x: feature, y: label
        """
        dataset = TFRecordDataset(filename)
        dataset = dataset.map(self.parser, num_threads=4, output_buffer_size=2048)
        # dataset = dataset.shuffle(buffer_size=1024, seed=42)
        dataset = dataset.repeat(count=-1)
        # dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        x, y = iterator.get_next()
        x, y = tf.train.shuffle_batch(
            tensors=[x, y], shapes=[list(self.shape), [self.num_labels]],
            batch_size=self.batch_size,
            capacity=2048,
            min_after_dequeue=1024,
            enqueue_many=False,
            num_threads=self.num_threads
        )
        return x, y

    @threadsafe_generator
    def generator(self, identifier='train'):
        if identifier == 'train':
            x_batch, y_batch = self.train_x_batch, self.train_y_batch
        elif identifier == 'validation':
            x_batch, y_batch = self.validation_x_batch, self.validation_y_batch
        else:
            x_batch, y_batch = self.test_x_batch, self.test_y_batch

        while not self.coordinator.should_stop():
            yield K.get_session().run([x_batch, y_batch])

    def transform_input(self, inp):
        if self.model_type == 'dnn':
            return K.flatten(inp)
        elif self.model_type == 'cnn':
            return K.expand_dims(inp, -1)
        else:
            return inp

    def clean_up(self):
        self.coordinator.request_stop()
        self.coordinator.join(self.threads)

    def get_complete_data(self, filename):
        dataset = TFRecordDataset(filename)
        dataset = dataset.map(self.parser)
        dataset = dataset.shuffle(buffer_size=1024, seed=42)
        iterator = dataset.make_one_shot_iterator()
        xt, yt = iterator.get_next()
        xs, ys = [], []
        while True:
            try:
                x, y = K.get_session().run([xt, yt])
                xs.append(x)
                ys.append(y)
            except:
                break
        return np.array(xs), np.array(ys)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    initialize_logger(None)

    parser = argparse.ArgumentParser(description='Run sample baseline models on TIMIT/VCTK data')
    parser.add_argument('train_tfrecord_file', type=str, help="Train TfRecord file path")
    parser.add_argument('validation_tfrecord_file', type=str, help="Validation TfRecord file path")
    parser.add_argument('test_tfrecord_file', type=str, help="Test TfRecord file path")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--model_type', type=str, default='dnn', help="Type of model to run",
                        choices=['dnn', 'rnn', 'cnn'])

    args = parser.parse_args()

    bdg = SampleDataGenerator(
        args.train_tfrecord_file,
        args.validation_tfrecord_file,
        args.test_tfrecord_file,
        args.batch_size, args.model_type
    )
    bdg.get_data_info()
