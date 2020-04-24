from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node
import random


class RandomSearch(TreeSearch):
    """
    Perfom a random search in the tree
    """
    def __init__(self, root: Node, evaluation_fn=lambda x: 0, n_samples=1, batch_size=None):
        TreeSearch.__init__(self, root, evaluation_fn)
        self.__path = None
        self.n_samples = n_samples
        self.batch_size = batch_size

    def random_expansion(self) -> Node:
        node = self.root
        self.__path = [node]
        while True:
            if node.is_terminal():
                return node
            node = random.choice(node.childrens())
            self.__path.append(node)

    def search(self) -> Node:
        best_node = None
        best_node_value = -1
        for i in range(self.n_samples):
            new_sample = self.random_expansion()
            new_sample_value = self._eval_node(new_sample)
            if new_sample_value >= best_node_value:
                best_node = new_sample
                best_node_value = new_sample_value
        return best_node

    def path(self):
        return self.__path
