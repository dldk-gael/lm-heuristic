from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node, Counter


class MonteCarloTreeSearch(TreeSearch):
    def __init__(self, root: Node, evaluation_fn, nb_of_tree_walks=1, batch_size=1):
        TreeSearch.__init__(self, root, evaluation_fn)
        self.nb_of_tree_walks = nb_of_tree_walks
        self.batch_size = batch_size
        self.counter_root = Counter(identifiant=hash(root), parent=None, is_terminal=root.is_terminal())

    def search(self):
        # perform the tree walks
        for i in range(self.nb_of_tree_walks):
            self.tree_walk()

        # follow the most visited path
        node = self.counter_root
        while not node.is_terminal():
            node = node.most_visited_children()
        return node

    def tree_walk(self):
        return

    def bandit_walk(self):


    def random_walk(self, source_node: Node):
        node = source_node
        while not node.is_terminal():
            node = source_node.random_children()
        return node

    def path(self):
        return