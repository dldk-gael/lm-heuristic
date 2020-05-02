from tree_search.strategy.search import TreeSearch
from tree_search.tree import CounterNode, Node
from typing import *

import math
from tqdm.autonotebook import tqdm
from time import time


class MonteCarloTreeSearch(TreeSearch):
    """
    Implement a Single Player Monte Carlo Tree Search as desbribe in
    Single-Player Monte-Carlo Tree Search for SameGame by Schadda, Winandsan, Taka, Uiterwijka in 2011

    Add a modification in order to be able to evaluate leaf node by batch
    """

    def __init__(
        self,
        root: Node,
        evaluation_fn: Callable[[List[Node]], List[float]],
        batch_size: int = 1,
        nb_of_tree_walks: int = 1,
        c: int = 1,
        d: int = 1000,
        t: int = 0,
    ):
        """
        Initialize the MCTS paramater and create the counter object that will be use to perform the MCTS
        :param root: Node object from which to perform the tree search
        :param evaluation_fn: scoring function that evaluate node (must be able to evaluate list of nodes in batch)
        :param batch_size: number of leaf to store in memory before evaluating them in one pass
        :param nb_of_tree_walks: number of tree walks to compute before selecting an action
        :param c: hyperparameter for upper confidence bound, control the exploration vs exploitation ratio
        :param d: hyperparameter for upper confidence bound
        :param t: threshold for expansion policy (see expansion_policy method)
        """
        assert (
            nb_of_tree_walks > t
        ), "You give a lower number of tree walks that threshold t : the root node will not be able to expand"
        TreeSearch.__init__(self, root, evaluation_fn)
        self.nb_of_tree_walks = nb_of_tree_walks
        self.batch_size = batch_size
        self.c = c
        self.d = d
        self.t = t

        # Wrap the root node in a counter object in order to maintain statistic on the path and rewards
        self.counter_root = CounterNode(reference_node=root, parent=None)
        print(
            "Initialize Monte Carlo Tree search \n"
            "\tthe parameters are C = %d, D= %d, t= %d\n"
            "\tit will perform %d of tree walks at each step"
            % (self.c, self.d, self.t, self.nb_of_tree_walks)
        )

        self.__path = []
        self.__time = 0
        self.total_nb_of_walks = 0
        self.search_result = None

    def search(self) -> Node:
        """
        Given the root nodes and all the parameters, search for the best leaf
        As describe in section 4.2 of 'Single-Player Monte-Carlo Tree Search for SameGame',
        we allocate ressorce move by move rather than using all the ressources from the tree node.

        Concretely :
        1. Compute nb_of_tree_walks by batch of batch_size
        2. Select the best node of the root
        3. Re-start the search from this node
        """
        current_root = self.counter_root

        begin_time = time()
        self.total_nb_of_walks = 0
        self.__path = [current_root]

        progress_bar = tqdm(
            total=self.nb_of_tree_walks, desc="Tree walks", unit="walks"
        )
        while not current_root.reference_node.is_terminal():
            progress_bar.reset()
            progress_bar.set_postfix(root_node=str(current_root.reference_node))

            # Perform nb of tree walks
            for _ in range(self.nb_of_tree_walks // self.batch_size):
                if current_root.solved:
                    break
                self.batch_tree_walks(current_root, self.batch_size)
                progress_bar.update(self.batch_size)
            if self.nb_of_tree_walks % self.batch_size != 0 and not current_root.solved:
                self.batch_tree_walks(
                    current_root, self.nb_of_tree_walks % self.batch_size
                )

            # Choose the best node among the childrens and continue the search from here
            current_root.freeze = True  # To avoid modifying the counter of previous roots in futur backpropagations
            current_root = current_root.top_children()
            self.__path.append(current_root)
            self.total_nb_of_walks += self.nb_of_tree_walks

        self.__time = time() - begin_time
        self.search_result = current_root

        return current_root.reference_node

    def batch_tree_walks(self, current_root: CounterNode, nb_tree_walks: int):
        """
        1. Perform batch_size tree walks keeping in memory at each time :
            - the terminal node from the counter tree (from which a random walk has been launched)
            - the terminal node from the root tree (the node at the end of the random walk)
        2. Evaluate all the terminal node from the root tree
        3. Backpropagate the reward informations in the counter tree
        """
        buffer = []
        for _ in range(nb_tree_walks):
            if current_root.solved:
                break
            buffer.append(self.single_tree_walk(current_root))

        nodes = [x[0] for x in buffer]
        leafs = [x[1] for x in buffer]
        rewards = self.evaluation_fn(leafs)
        for node, leaf, reward in zip(nodes, leafs, rewards):
            node.backpropagate(reward, leaf)

    def single_tree_walk(self, current_root: CounterNode) -> (CounterNode, Node):
        """
        Perform a single tree walk.
        1. The 'bandit phase' / selection step: if current node has been explored so far,
            we choose as a next node the children node with the maximum upper bound confidence value
        2. The expension step: once at the frontier of the counter tree, we randomly choose one new node to expand
        3. The play-out step: from there we go randomly to the end of the tree
        """
        # bandit phase using selection policy
        counter_node = current_root
        counter_node.count += 1
        while not counter_node.is_terminal():
            counter_node = self.selection_policy(counter_node)
            counter_node.count += 1

        # expansion phase
        # Expand a node only if he has been visited t times so far as desbribe in
        # Coulom, R., 2007. Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
        if (
            not counter_node.reference_node.is_terminal()
            and counter_node.count + 1 > self.t
        ):
            counter_node.expand()

        if counter_node.reference_node.is_terminal():
            counter_node.set_as_solved()

        # perform a random walk from there
        random_leaf = counter_node.reference_node.random_walk()

        return counter_node, random_leaf

    def selection_policy(self, counter_node: CounterNode) -> CounterNode:
        """
        if a children of current node has been visited yet: visit it
        if all children has been visited already one: select the children with the biggest upper bound confidence
        """
        for children in counter_node.childrens():
            if children.count == 0:
                return children

        total_selection = counter_node.count

        # Select node that maximise UPC + skip node that have been solved
        unsolved_childrens = [
            children for children in counter_node.childrens() if not children.solved
        ]
        assert len(unsolved_childrens) > 0, "Try to select from solved node"

        return max(
            unsolved_childrens,
            key=lambda node: self.node_upper_confidence_bound(node, total_selection),
        )

    def node_upper_confidence_bound(
        self, node: CounterNode, total_nb_of_selections: int
    ) -> float:
        """
        Compute the upper confidence bound as describe in section 4.1 of
        'Single-Player Monte-Carlo Tree Search for SameGame'
        """
        return (
            node.sum_rewards / node.count
            + math.sqrt(self.c * math.log(total_nb_of_selections / node.count))
            + math.sqrt(
                (
                    node.sum_of_square_rewards
                    - node.count * ((node.sum_rewards / node.count) ** 2)
                    + self.d
                )
                / node.count
            )
        )

    def path(self) -> List[Node]:
        return list(map(lambda counter_node: counter_node.reference_node, self.__path))

    def counter_path(self) -> List[CounterNode]:
        return self.__path  # useful for analyse / debug

    def search_info(self) -> Dict:
        assert (
            self.search_result is not None
        ), "try to access search info but no search was computed"
        return {
            "time": self.__time,
            "path": self.path(),
            "total_nb_of_walks": self.total_nb_of_walks,
            "best_leaf": self.search_result.reference_node,
            "best_leaf_value": self._eval_node([self.search_result.reference_node])[0],
        }
