import multiprocessing
import time 
import queue
from lm_heuristic.utils.timer import TimeComputation

def worker(to_eval_queue, result_queue):
    while True:
        to_eval = to_eval_queue.get(block=True)
        result_queue.put("Result")

if __name__ == '__main__':
    to_eval_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(to_eval_queue, result_queue))
    p.start()
    time.sleep(2)
    with TimeComputation("Send/Recieve 1 Msg") :
        to_eval_queue.put("Task 1")
        result = result_queue.get()

    p.kill()    