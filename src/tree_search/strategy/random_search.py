from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node
import random


class RandomSearch(TreeSearch):
    def __init__(self, root: Node):
        TreeSearch.__init__(self, root)
        self.root = root
        self.path = []

    def search(self) -> Node:
        node = self.root
        self.path = [node]
        while True:
            if node.is_terminal():
                return node
            node = random.choice(node.childrens())
            self.path.append(node)
