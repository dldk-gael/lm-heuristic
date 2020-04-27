from tree_search.strategy.search import TreeSearch
from tree_search.tree import Node, Counter

import math


class MonteCarloTreeSearch(TreeSearch):
    def __init__(self, root: Node, evaluation_fn, nb_of_tree_walks=1, batch_size=1, ce=1):
        TreeSearch.__init__(self, root, evaluation_fn)
        self.nb_of_tree_walks = nb_of_tree_walks
        self.batch_size = batch_size
        self.counter_root = Counter(reference_node=root, parent=None)
        self.ce = ce

    def search(self):
        # perform the tree walks
        for i in range(self.nb_of_tree_walks // self.batch_size):
            self.batch_tree_walks()

        # follow the most visited path
        node = self.counter_root
        while not node.is_terminal():
            node = node.most_visited_children()
        return node

    def single_tree_walk(self):
        # bandit phase
        counter_node = self.counter_root
        while not counter_node.is_terminal():
            counter_node = self.bandit_policy(counter_node.childrens())

        # grow a leaf
        counter_node.expand()
        new_counter_node = counter_node.random_children()
        reference_node = new_counter_node.reference_node
        # perform the random walk for the new node
        return new_counter_node, reference_node.random_walk()

    def batch_tree_walks(self):
        buffer = [self.single_tree_walk() for _ in range(self.batch_size)]
        nodes = [x[0] for x in buffer]
        rewards = self.evaluation_fn([x[1] for x in buffer])
        for node, reward in nodes, rewards:
            self.backpropagate(node, reward)

    def backpropagate(self, counter_node, reward):
        counter_node.update(reward)
        if counter_node.parent is not None:
            self.backpropagate(counter_node.parent, reward)

    def confidence_bound(self, nb_of_times_selected, total_nb_of_selections, average_reward):
        return average_reward + math.sqrt(self.ce * math.log(total_nb_of_selections / nb_of_times_selected))

    def bandit_policy(self, counter_nodes_list):
        total_nb_of_selections = sum(counter_node.count for counter_node in counter_nodes_list)
        node_confidence_bound = lambda counter_node: self.confidence_bound(counter_node.count,
                                                                           total_nb_of_selections,
                                                                           counter_node.mean_reward)

        return max(counter_nodes_list, key=node_confidence_bound)

    def path(self):
        return