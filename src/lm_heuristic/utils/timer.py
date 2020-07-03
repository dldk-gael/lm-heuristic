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
    def wrapper(*args, **kwargs):
        begin_time = time.process_time()
        res = func(*args, **kwargs)
        end_time = time.process_time()
        wrapper.time_spent += end_time - begin_time
        return res

    wrapper.time_spent = 0.0
    return wrapper

######################################################################
## Other time utility function
######################################################################

def print_timer(func):
    print("Time spent in %s : %0.2f ms" % (func.__name__, func.time_spent * 1000))
