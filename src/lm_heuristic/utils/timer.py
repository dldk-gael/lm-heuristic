"""
Define various tools to keep track of times
"""

import time
from functools import wraps

import torch


######################################################################
## Define context manager that keep track of the time spent inside
######################################################################

class TimeGPUComputation:
    def __init__(self, step_name):
        self.step_name = step_name
        assert torch.cuda.is_available(), "Try to track GPU computation but cuda is not available"

    def __enter__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()

    def __exit__(self, *args, value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print("%s : %f ms" % (self.step_name, elapsed_time_ms))


class TimeComputation:
    """
    Context manager that compute time spent inside
    """

    def __init__(self, step_name):
        self.step_name = step_name

    def __enter__(self):
        self.begin_time = time.process_time()

    def __exit__(self, *args):
        end_time = time.process_time()
        elapsed_time_ms = (end_time - self.begin_time) * 1000
        print("%s : %f ms" % (self.step_name, elapsed_time_ms))


######################################################################
## Define decorators that keep track of the time spent inside 
## a function
######################################################################

def time_function(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        begin_time = time.process_time()
        res = func(self, *args, **kwargs)
        end_time = time.process_time()
        self._timer.setdefault(func.__name__, 0.0)
        self._timer[func.__name__] += end_time - begin_time
        return res
    
    return wrapper

class Timer:
    def __init__(self):
        self._timer = dict()
    
    def reset_timer(self):
        for key in self._timer:
            self._timer[key] = 0.0 
    
    def print_timer(self):
        for key in self._timer: 
            print("Time spent in %s : %0.2f ms" % (key, self._timer[key] * 1000))

