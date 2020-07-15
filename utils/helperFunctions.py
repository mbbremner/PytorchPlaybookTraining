import os
import io
import json
import math
import numpy as np
import time


def make_directory(path):
    """ Create a directory which corresponds to the link file for a
        given Bing query"""
    created = False
    if not os.path.isdir(path):
        os.makedirs(path)
        print("Directory ", path, " Created ")
        created = True
    else:
        print("\t Directory %s, already exists" % path)

    return created


def openReadLines(path):
    """
    Read a single file into lines
    :param path:
    :return:
    """
    with io.open(path, encoding='utf-8') as f:
        lines = f.readlines()
        f.close()
    return lines


def saveTupleList(tuple_list, path):
    with io.open(path, 'w', encoding='utf-8') as f:
        f.writelines([','.join([str(item) for item in tuple]) + '\n' for tuple in tuple_list])


def openManyFiles(root, fileList):

    all_data = []
    for f, file in enumerate(fileList):
        with io.open(root + file, encoding='utf-8') as f:
            data = f.read()
            f.close()
        all_data.append(data)
    return all_data


def jsonLoad(path):
    """
    Load a single json file into a dictionary object
    :param path: path to json file
    :return: dictionary object
    """
    with open(path, encoding='utf-8') as json_file:
        json_dict = json.load(json_file)
        json_file.close()
    return json_dict


def count_extensions(directory, ext):
    count = 0
    file_names = os.listdir(directory)
    for file in file_names:
        if os.path.splitext(file)[1] == ext:
            count += 1
    return count


def count_num_correct(labs, preds):
    correct = 0
    for p, pred in enumerate(preds):
        if pred == labs[p]:
            correct += 1
    return correct

# decorator to calculate duration
# taken by any function.
def calculate_time(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()
        func(*args, **kwargs)

        # storing time after function execution
        end = time.time()
        print("\t-- Execution time for %s:  %3.3f seconds --" % (func.__name__, end - begin))

    return inner1


def compute_precision_vals(cm):
    """ Compute the precision values for a confusion matrix"""
    cm = cm.T
    recalls = []
    for r, row in enumerate(cm):
        recalls.append(precision(r, row))
    return recalls


def precision(i, row):
    tp = 0
    fp = 0
    precisionVal = 0
    for n, num in enumerate(row):
        if n == i:
            tp += num
        else:
            fp += num
    if tp != 0:
        precisionVal = tp / (tp + fp)

    return precisionVal


def inner_update_wrap(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner_func(*args, **kwargs):
        # storing time before function execution

        func(*args, **kwargs)

        # storing time after function execution


    return inner_func