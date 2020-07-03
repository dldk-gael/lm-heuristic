import threading
import time 
import queue
from lm_heuristic.utils.timer import TimeComputation

a = []
class Worker(threading.Thread):
    def __init__(self, tasks_queue, results_queue):
        threading.Thread.__init__(self)
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

    def run(self):
        while True:
            print("Worker: waiting task")
            task = self.tasks_queue.get(block=True)
            a.append("FROM THREAD")
            print("Worker: recieve task")
            self.results_queue.put("Result")
    


if __name__ == '__main__':
    tasks_queue = queue.Queue()
    results_queue = queue.Queue()
    worker = Worker(tasks_queue, results_queue)
    worker.daemon = True
    worker.start()
    time.sleep(2)

    with TimeComputation("Send/Recieve 1 Msg"):
        print("Master: sending task")
        tasks_queue.put("Task 1")
        print("Master: recieve results")
        result = results_queue.get(block=True)

    print(result)
    print(a)
