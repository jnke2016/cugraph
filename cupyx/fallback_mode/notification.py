"""
Methods related to notifications.
"""

import warnings
import threading

_thread_locals = threading.local()
_thread_locals.dispatch_type = 'warn'

FallbackWarning = type('FallbackWarning', (Warning,), {})
warnings.simplefilter(action='always', category=FallbackWarning)


def geterr():
    return _thread_locals.dispatch_type


def seterr(new_dispatch):

    old = _thread_locals.dispatch_type

    if new_dispatch in ['print', 'warn', 'log', 'ignore', 'raise']:
        _thread_locals.dispatch_type = new_dispatch
        return old

    raise ValueError('{} is not valid dispatch type'.format(new_dispatch))


class FallbackLogger:

    logger = None

    @classmethod
    def setlogger(cls, logger):
        cls.logger = logger

    @classmethod
    def getlogger(cls):

        if cls.logger is None:
            raise AttributeError('Logger not initiated')

        return cls.logger


def dispatch_notification(func):

    msg = "'{}' method not in cupy, falling back to '{}.{}'".format(
        func.__name__, func.__module__, func.__name__)

    raise_msg = "'{}' method not found in cupy".format(
        func.__name__)

    if _thread_locals.dispatch_type == 'print':
        print("Warning: {}".format(msg))

    elif _thread_locals.dispatch_type == 'warn':
        warnings.warn(msg, FallbackWarning, stacklevel=3)

    elif _thread_locals.dispatch_type == 'ignore':
        pass

    elif _thread_locals.dispatch_type == 'log':
        logger = FallbackLogger.getlogger()
        logger.warning(msg)

    elif _thread_locals.dispatch_type == 'raise':
        raise AttributeError(raise_msg)

    else:
        assert False


class errstate:

    def __init__(self, new_dispatch):
        self.old = None
        self.new = new_dispatch

    def __enter__(self):
        self.old = _thread_locals.dispatch_type
        seterr(self.new)

    def __exit__(self, *exc_info):
        seterr(self.old)
