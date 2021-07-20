# -*- coding: utf-8 -*-
import sys
from os.path import dirname, abspath
from config import FLAGS
import tensorflow as tf
import networkx as nx
import numpy as np

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../src'.format(cur_folder))

def check_flags():
    # assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 2)
    assert (FLAGS.batch_size >= 1)
    assert (FLAGS.iters >= 1)
    assert (FLAGS.iters_val_start >= 1)
    assert (FLAGS.gpu >= -1)
    d = FLAGS.flag_values_dict()
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))


def convert_long_time_to_str(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} days {} hours {} mins {:.1f} secs'.format(
        int(day), int(hour), int(minutes), seconds)


def get_modeL_info_as_str(model_info_table=None):
    rtn = []
    d = FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)


def load(filepath):
    from os.path import isfile
    filepath = proc_filepath(filepath)
    if isfile(filepath):
        with open(filepath, 'rb') as handle:
            return load_pkl(handle)

def load_joblib(filepath):
    from sklearn.externals import joblib
    from os.path import isfile
    filepath = proc_filepath(filepath)
    if isfile(filepath):
        return joblib.load(filepath)


def save(filepath, obj):
    with open(proc_filepath(filepath), 'wb') as handle:
        save_pkl(obj, handle)

def save_joblib(filepath, obj):
    from sklearn.externals import joblib
    joblib.dump(obj, filepath)

def save_npy(filepath, obj):
    import numpy as np
    with open(proc_filepath_npy(filepath), 'wb') as handle:
        np.save(handle, obj)

def proc_filepath_npy(filepath):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.npy'
    if ext not in filepath:
        filepath += ext
    return filepath


def proc_filepath(filepath):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.pickle'
    if ext not in filepath:
        filepath += ext
    return filepath


def load_pkl(handle):
    import pickle
    return pickle.load(handle, encoding='iso-8859-1')

def save_pkl(obj, handle):
    import pickle
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_save_path():
    return get_root_path() + '/save'


def get_data_path():
    return get_root_path() + '/data'


def get_result_path():
    return get_root_path() + '/result'


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def get_model_path():
    return get_root_path() + '/model'


def get_file_base_id(file):
    return int(file.split('/')[-1].split('.')[0])


def get_train_str(train_bool):
    if train_bool == True:
        return 'train'
    elif train_bool == False:
        return 'test'
    else:
        assert (False)


def load_data(data, train):
    if data == 'aids700nef':
        from data import AIDS700nefData
        return AIDS700nefData(train)
    elif data == 'linux':
        from data import LinuxData
        return LinuxData(train)
    elif data == 'imdbmulti':
        from data import IMDBMultiData
        return IMDBMultiData(train)
    elif data == 'mivia':
        from data import MiviaData
        return MiviaData(train)
    else:
        raise RuntimeError('Not recognized data %s' % data)


def get_norm_str(norm):
    if norm is None:
        return ''
    elif norm:
        return '_norm'
    else:
        return '_nonorm'


def create_dir_if_not_exists(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


exec_print = True


def exec_turnoff_print():
    global exec_print
    exec_print = False


def exec_turnon_print():
    global exec_print
    exec_print = True


def exec(cmd, timeout=None):
    global exec_print
    if not timeout:
        from os import system
        if exec_print:
            print(cmd)
        else:
            cmd += ' > /dev/null'
        system(cmd)
        return True  # finished
    else:
        import subprocess as sub
        import threading

        class RunCmd(threading.Thread):
            def __init__(self, cmd, timeout):
                threading.Thread.__init__(self)
                self.cmd = cmd
                self.timeout = timeout

            def run(self):
                self.p = sub.Popen(self.cmd, shell=True)
                self.p.wait()

            def Run(self):
                self.start()
                self.join(self.timeout)

                if self.is_alive():
                    self.p.terminate()
                    self.join()
                    self.finished = False
                else:
                    self.finished = True

        if exec_print:
            print('Timed cmd {}sec {}'.format(timeout, cmd))
        r = RunCmd(cmd, timeout)
        r.Run()
        return r.finished


tstamp = None


def get_ts():
    import datetime
    import pytz
    global tstamp
    if not tstamp:
        tstamp = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime('%Y-%m-%dT%H:%M:%S')
    return tstamp


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def get_flags(k):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        return None


def format_float(f):
    if f < 1e-2 and f != 0:
        return '{:.3e}'.format(f)
    else:
        return '{:.3f}'.format(f)


def convert_msec_to_sec_str(sec):
    return '{:.2f}msec'.format(sec * 1000)


def get_model_info_as_str(model_info_table=None):
    rtn = []
    d = FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def need_val(iter):
    assert (iter != 0)  # 1-based iter
    return (iter == 1 or \
            iter >= FLAGS.iters_val_start and iter % FLAGS.iters_val_every == 0)


def prompt(str, options=None):
    while True:
        t = input(str + ' ')
        if options:
            if t in options:
                return t
        else:
            return t


def prompt_get_computer_name():
    global computer_name
    if not computer_name:
        computer_name = prompt('What is the computer name?')
    return computer_name


def get_siamese_dir():
    return cur_folder


def save_as_dict(filepath, *args, **kwargs):
    '''
    Warn: To use this function, make sure to call it in ONE line, e.g.
    save_as_dict('some_path', some_object, another_object)
    Moreover, comma (',') is not allowed in the filepath.
    '''
    import inspect
    from collections import OrderedDict
    frames = inspect.getouterframes(inspect.currentframe())
    frame = frames[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    dict_to_save = OrderedDict()
    all_args_strs = string[string.find('(') + 1:-1].split(',')
    if 1 + len(args) + len(kwargs) != len(all_args_strs):
        msgs = ['Did you call this function in one line?',
                'Did the arguments have comma "," in the middle?']
        raise RuntimeError('\n'.join(msgs))
    for i, name in enumerate(all_args_strs[1:]):
        if name.find('=') != -1:
            name = name.split('=')[1]
        name = name.strip()
        if i >= 0 and i < len(args):
            dict_to_save[name] = args[i]
        else:
            break
    dict_to_save.update(kwargs)
    print('Saving a dictionary as pickle to {}'.format(filepath))
    save(filepath, dict_to_save)


def load_as_dict(filepath):
    print('Loading a dictionary as pickle from {}'.format(filepath))
    return load(filepath)


def check_nx_version():
    import networkx as nx
    nxvg = '1.10'
    nxva = nx.__version__
    if nxvg != nxva:
        raise RuntimeError( \
            'Wrong networkx version! Need {} instead of {}'.format(nxvg, nxva))


def prompt_get_cpu():
    from os import cpu_count
    while True:
        num_cpu = prompt( \
            '{} cpus available. How many do you want?'.format( \
                cpu_count()))
        num_cpu = parse_as_int(num_cpu)
        if num_cpu and num_cpu <= cpu_count():
            return num_cpu


def parse_as_int(s):
    try:
        rtn = int(s)
        return rtn
    except ValueError:
        return None
