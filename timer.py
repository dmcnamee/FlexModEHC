#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://gist.github.com/bradmontgomery/bd6288f09a24c06746bbe54afe4b8a82

from functools import wraps
import logging
import time
import config

logger = logging.getLogger('__SGS__')
if config.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def timeit_debug(func):
    """This decorator prints the execution time for the decorated function."""
    if config.timing and config.verbose:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.debug(" {} ran in {}s".format(func.__name__, round(end - start, 5)))
            return result
        return wrapper
    else:
        return func

def timeit_info(func):
    """This decorator prints the execution time for the decorated function."""
    if config.timing:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.info(" {} ran in {}s".format(func.__name__, round(end - start, 5)))
            return result
        return wrapper
    else:
        return func
