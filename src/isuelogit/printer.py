# -*- coding: utf-8 -*-

"""Console communication """

import time
import sys, os
from contextlib import contextmanager
import io

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, eraseBar = True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    # https: // stackoverflow.com / questions / 3173320 / text - progress - bar - in -the - console

    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    # filledLength = int(length * iteration // total)
    # bar = fill * filledLength + '-' * (length - filledLength)

    # unicode character '█' only works in Linux/mac environments
    # bar_symbol = '█'
    bar_symbol = '#'

    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(length * iteration / float(total)))

    bar = bar_symbol * filled_length + '-' * (length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    # print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)

    # When Completed
    if iteration == total:
        if eraseBar:
            # print('', end='\r')
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

    sys.stdout.flush()

def printIterationBar(iteration, iterations, prefix = '', length = 100, eraseBar = False):
    """
    Call in a loop to create terminal progress bar
    """

    # https: // stackoverflow.com / questions / 3173320 / text - progress - bar - in -the - console

    # unicode character '█' only works in Linux/mac environments
    #bar_symbol = '█'
    bar_symbol = '#'

    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(round(length * iteration / float(iterations)))
    bar = bar_symbol * filled_length + '-' * (length - filled_length)
    iteration = '{:d}'.format(iteration)
    iterations = '{:d}'.format(iterations)

    # print(f'\r{prefix} |{bar}| {iteration}/{iterations}', end = printEnd)

    sys.stdout.write('\r%s |%s| %s%s%s ' % (prefix, bar, iteration, '/', iterations)),

    # When Completed
    if iteration == iterations:
        if eraseBar:
            # print('', end='\r')
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')

    sys.stdout.flush()


def timeit(method):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

def blockPrint():
    pass

    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # sys.__stdout__ = sys.stdout
    # sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    pass
    # sys.stdout = sys.__stdout__


# decorator used to block function printing to the console
def blockPrinting(func):

    def func_wrapper(*args, **kwargs):
        sys.__stdout__ = sys.stdout
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper

@contextmanager
def block_output(show_stdout=True, show_stderr=True):
    suppress_stdout, suppress_stderr = not show_stdout, not show_stderr,

    # https://www.codeforests.com/2020/11/05/python-suppress-stdout-and-stderr/

    stdout = sys.stdout
    stderr = sys.stderr
    devnull = open(os.devnull, "w")
    try:
        if suppress_stdout:
            sys.stdout = devnull
        if suppress_stderr:
            sys.stderr = devnull
        yield
    finally:
        if suppress_stdout:
            sys.stdout = stdout
        if suppress_stderr:
            sys.stderr = stderr
