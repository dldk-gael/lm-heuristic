from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Node(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def childrens(self) -> List[Node]:
        pass

    @abstractmethod
    def __hash__(self):
        pass

