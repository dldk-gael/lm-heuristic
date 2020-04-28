from tree_search.strategy.search import TreeSearch
from tree_search.tree import Counter

import math
import logging
from tqdm.autonotebook import tqdm


logger = logging.getLogger(__name__)


class MonteCarloTreeSearch(TreeSearch):
    """
    Implement a Single Player Monte Carlo Tree Search as desbribe in
    Single-Player Monte-Carlo Tree Search for SameGame by Schadda, Winandsan, Taka, Uiterwijka in 2011

    Add a modification in order to be able to evaluate leaf node by batch
    """

    def __init__(self, root, evaluation_fn, batch_size=1, nb_of_tree_walks=1, c=1, d=1000, t=0):
        """
        Initialize the MCTS paramater and create the counter object that will be use to perform the MCTS
        :param root: Node object from which to perform the tree search
        :param evaluation_fn: scoring function that evaluate node (must be able to evaluate list of nodes in batch)
        :param batch_size: number of leaf to store in memory before evaluating them in one pass
        :param nb_of_tree_walks: number of tree walks to compute before selecting an action TODO: replace that by time
        :param c: hyperparameter for upper confidence bound, control the exploration vs exploitation ratio
        :param d: hyperparameter for upper confidence bound TODO : add more explanation here
        :param t: threshold for expansion policy (see expansion_policy method)
        """
        assert nb_of_tree_walks > t, (
            "You give a lower number of tree walks that threshold t : the root node will not be able to expand"
        )
        TreeSearch.__init__(self, root, evaluation_fn)
        self.nb_of_tree_walks = nb_of_tree_walks
        self.batch_size = batch_size
        self.c = c
        self.d = d
        self.t = t

        # Wrap the root node in a counter object in order to maintain statistic on the path and rewards
        self.counter_root = Counter(reference_node=root, parent=None)
        logging.info("Initialize Monte Carlo Tree search \n"
                     "\tthe parameters are C = %d, D= %d, t= %d\n"
                     "\tit will perform %d of tree walks at each step" %
                     (self.c, self.d, self.t, self.nb_of_tree_walks))

    def search(self):
        """
        Given the root nodes and all the parameters, search for the best leaf
        As describe in section 4.2 of 'Single-Player Monte-Carlo Tree Search for SameGame',
        we allocate ressorce move by move rather than using all the ressources from the tree node.

        Concretely :
        1. Compute nb_of_tree_walks
        2. Select the best node of the root
        3. Re-start the search from this node
        """
        while not self.counter_root.reference_node.is_terminal():
            logger.info("Launching tree walks from node : <%s>" % str(self.counter_root.reference_node))
            self.one_step()
            self.counter_root = self.counter_root.top_children()
        return self.counter_root.reference_node

    def one_step(self):
        """
        Perform nb_of_tree_walks by batch of batch_size
        """
        for _ in range(self.nb_of_tree_walks // self.batch_size):
            self.batch_tree_walks(self.batch_size)
        if self.nb_of_tree_walks % self.batch_size != 0:
            self.batch_tree_walks(self.nb_of_tree_walks % self.batch_size)

    def batch_tree_walks(self, batch_size):
        """
        1. Perform batch_size tree walks keeping in memory at each time :
            - the terminal node from the counter tree (from which a random walk has been launched)
            - the terminal node from the root tree (the node at the end of the random walk)
        2. Evaluate all the terminal node from the root tree
        3. Backpropagate the reward informations in the counter tree
        """
        buffer = [self.single_tree_walk() for _ in range(batch_size)]
        nodes = [x[0] for x in buffer]
        rewards = self.evaluation_fn([x[1] for x in buffer])
        for node, reward in zip(nodes, rewards):
            node.update_and_backpropagate(reward)

    def single_tree_walk(self):
        """
        Perform a single tree walk.
        1. The 'bandit phase' / selection step: if current node has been explored so far,
            we choose as a next node the children node with the maximum upper bound confidence value
        2. The expension step: once at the frontier of the counter tree, we randomly choose one new node to expand
        3. The play-out step: from there we go randomly to the end of the tree
        """
        # bandit phase using selection policy
        counter_node = self.counter_root
        while not counter_node.is_terminal():
            counter_node = self.selection_policy(counter_node)

        # expansion phase
        counter_node = self.expansion_policy(counter_node)

        # perform a random walk from there
        random_leaf = counter_node.reference_node.random_walk()

        return counter_node, random_leaf

    def selection_policy(self, counter_node):
        """
        if a children of current node has been visited yet: visit it
        if all children has been visited already one: select the children with the biggest upper bound confidence
        """
        for children in counter_node.childrens():
            if children.count == 0:
                return children

        total_selection = counter_node.count
        return max(counter_node.childrens(), key=lambda node: self.node_upper_confidence_bound(node, total_selection))

    def node_upper_confidence_bound(self, node, total_nb_of_selections):
        """
        Compute the upper confidence bound as describe in section 4.1 of
        'Single-Player Monte-Carlo Tree Search for SameGame'
        """
        return node.average_reward + \
               math.sqrt(self.c * math.log(total_nb_of_selections / node.count)) + \
               math.sqrt((node.sum_of_square_rewards - node.count * (node.average_reward ** 2) + self.d) / node.count)

    def expansion_policy(self, counter_node):
        """
        Expand a node only if he has been visited t times so far as desbribe in
        Coulom, R., 2007. Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search
        """
        if counter_node.reference_node.is_terminal() or counter_node.count < self.t:
            return counter_node
        else:
            counter_node.expand()
            return counter_node.random_children()

    def path(self):
        return
