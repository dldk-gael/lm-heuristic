import time
from functools import wraps


def timeit(func):
    """
    In order to evaluate the time spent in a class :
    - make the class inheritate from Timer
    - decorate the method you want to track time with @timeit
    :param func: method to decorate
    """

    def inner(self, *args, **kwargs):
        begin_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        self._time_spent += time.perf_counter() - begin_time
        return result

    return inner


class Timer:
    """
    Timer class allows to keep track of time spent in methods that are decorated with @timeit
    """

    def __init__(self):
        self._time_spent = 0

    def reset_timer(self):
        self._time_spent = 0

    def time_spent(self):
        return self._time_spent


## CONTEXT MANAGER THAT COMPUTE TIME SPENT INSIDE 

class TimeComputation:
    """
    Context manager that compute time spent inside 
    """
    def __init__(self, step_name):
        self.step_name = step_name 

    def __enter__(self):
        self.begin_time = time.process_time()

    def __exit__(self, type, value, traceback):
        end_time = time.process_time()
        elapsed_time_ms = (end_time - self.begin_time) * 1000
        print("%s : %f ms" % (self.step_name, elapsed_time_ms))


## DECORATOR THAT KEEP TRACK OF TIME SPENT INSIDE

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

def print_timer(func):
    print("Time spent in %s : %0.2f ms" % (func.__name__, func.time_spent * 1000))