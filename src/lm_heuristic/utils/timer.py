import time


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
