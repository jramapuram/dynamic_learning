import numpy as np
import tensorflow as tf

from collections import OrderedDict
from sklearn.preprocessing import Normalizer

# Add more functions here to auto gen models and make them part of the process
def _get_gpu_map():
    return OrderedDict(
        [
            ('x2', [lambda x: x, lambda x: tf.pow(x, 2)])
            , ('x3', [lambda x: x, lambda x: tf.pow(x, 3)])
            , ('sign', [lambda x: x, tf.sign])
            , ('floor', [lambda x: x, tf.floor])
            , ('sqrt', [lambda x: x, tf.sqrt])
            , ('cos', [lambda x: x, tf.cos])
            , ('-x4', [lambda x: x, lambda x: -tf.pow(x, 4)])
            , ('sin', [lambda x: x, tf.sin])
            , ('tanh', [lambda x: x, tf.nn.tanh])
            , ('abs', [lambda x: x, tf.abs])
        ])

# Add more functions here to auto gen models and make them part of the process
def _get_cpu_map():
    return OrderedDict(
        [
            ('x2', [lambda x: x, lambda x: np.power(x, 2)])
            , ('x3', [lambda x: x, lambda x: np.power(x, 3)])
            , ('sign', [lambda x: x, np.sign])
            , ('floor', [lambda x: x, np.floor])
            , ('sqrt', [lambda x: x, np.sqrt])
            , ('cos', [lambda x: x, np.cos])
            , ('-x4', [lambda x: x, lambda x: -np.power(x, 4)])
            , ('sin', [lambda x: x, np.sin])
            , ('tanh', [lambda x: x, np.tanh])
            , ('abs', [lambda x: x, np.abs])
        ])

regression_keys = [w for w in _get_cpu_map()]

def function_name_to_index(fname):
    switcher = {
        'x2'    : 0,
        'x3'    : 1,
        'sign'  : 2,
        'floor' : 3,
        'sqrt'  : 4,
        'cos'   : 5,
        '-x4'   : 6,
        'sin'   : 7,
        'tanh'  : 8,
        'abs'   : 9
    }
    return switcher.get(fname, lambda: -1)

# Generate a wave and window it based on input_size
class RegressionGenerator(object):
    def __init__(self, function, input_size, sess=None, sequential=False):
        self.input_size = input_size
        self.function = function
        self.sequential = sequential
        self.sess = sess
        self.delta = 1.0/(input_size - 1.0)
        self.data_map = _get_cpu_map() if sess is None else _get_gpu_map()

    def _generate_generic_wave(self, func, input_size, batch_size):
        if not self.sequential:
            rng = np.random.RandomState(1)
            indexes = np.sort(10* rng.rand(input_size, batch_size), axis=0).T
            #indexes = np.random.uniform(low=0.0, high=1.0, size=[batch_size, input_size])
        else:
            indexes = np.linspace(self.delta
                        , batch_size*2*np.pi + self.delta
                        , num=input_size*batch_size).reshape([batch_size, input_size])
            self.delta = self.delta + 1.0/(input_size - 1.0) \
                   if self.delta < 2.0 * np.pi else  1.0/(input_size - 1.0)

        if self.sess is None:
            io_funcs = self.data_map[func]
            input_wave = np.asarray(io_funcs[0](indexes), dtype=np.float32)
            target_wave = np.asarray(io_funcs[1](indexes), dtype=np.float32)
            tmean = np.mean(target_wave)
            tvar = np.var(target_wave)

            return [input_wave, (target_wave - tmean)/(tvar + 1e-9)]
        else:
            io_funcs = self.data_map[func]
            input_wave = io_funcs[0](indexes)
            target_wave = io_funcs[1](indexes)
            #return self.sess.run([unit_scale(input_wave), unit_scale(target_wave)])
            #return self.sess.run([tf.nn.l2_normalize(input_wave, dim=1), tf.nn.l2_normalize(target_wave, dim=1)])
            # return self.sess.run([tf.to_float(tf.nn.l2_normalize(input_wave, dim=1))
            #             , tf.to_float(tf.nn.l2_normalize(target_wave, dim=1))])
            return self.sess.run([tf.to_float(input_wave), tf.to_float(target_wave)])


    def get_test_batch_iter(self, batch_size):
        return self._generate_generic_wave(self.function, self.input_size, batch_size)

    def get_batch_iter(self, batch_size):
        return self._generate_generic_wave(self.function, self.input_size, batch_size)
