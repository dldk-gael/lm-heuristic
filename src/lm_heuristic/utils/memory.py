from typing import *


class Memory:
    def __init__(self):
        self._memory = dict()
        self._call_history = dict()
        self._best_key = None
        self._best_value = -1.0

    def reset(self):
        self._memory = dict()
        self._call_history = list()
        self._best_key = None
        self._best_value = -1.0

    def has_already_eval(self, key):
        return key in self._memory

    def value_from_memory(self, key):
        value = self._memory[key]
        self._call_history.append((key, value))
        return value

    def update_memory(self, key_value_list):
        for key, value in key_value_list:
            self._call_history.append((key, value))
            self._memory[key] = value
            if value > self._best_value:
                self._best_key, self._best_value = key, value

    def best_in_memory(self):
        return self._best_key, self._best_value

    def top_n_best(self, top_n):
        return sorted(self._memory.items(), key=lambda x: x[1], reverse=True)[:top_n]

    def history_keys(self):
        return list(self._call_history.keys())

    def history_values(self):
        return list(self._call_history.values())