import math
from typing import List, Tuple, Dict

from lm_heuristic.heuristic import Heuristic
from lm_heuristic.tree import CounterNode, Node, TreeStats
from lm_heuristic.tree_search import TreeSearch
from .allocation_strategy import AllocationStrategy, RessourceAllocation


class MonteCarloTreeSearch(TreeSearch):
    """
    Searcher using a MCTS strategy, it mainly follows the ideas describe in
    Single-Player Monte-Carlo Tree Search for SameGame by Schadda, Winandsan, Taka, Uiterwijka

    Some other improvments have been implemented :
    - possibility to store the tree's leaves in buffer to evaluate them by batch
    - mark the node that have been solved to accelerate the exploration
        (ie all terminal childrens that can been access from this node has been evaluated once)
    """

    def __init__(
        self,
        heuristic: Heuristic,
        buffer_size: int = 1,
        c: int = 1,
        d: int = 1000,
        t: int = 0,
        stats_samples: int = 100,
        allocation_strategy: AllocationStrategy = AllocationStrategy.UNIFORM,
        verbose: bool = False,
        name: str = "MCTS",
    ):
        """
        Initialize the MCTS paramater and create the counter object that will be use to perform the MCTS
        :param heuristic: Heuristic instance
        :param buffer_size: number of leaf to store in memory before evaluating them in one pass
        :param c: hyperparameter for upper confidence bound, control the exploration vs exploitation ratio
        :param d: hyperparameter for upper confidence bound
        :param t: threshold for expansion policy (see expansion_policy method)
        :param stats_samples: number of tree walks used to assess tree size
        :param allocation_strategy: strategy that determine how many tree walks will be performed at each layer
        """
        TreeSearch.__init__(self, heuristic, buffer_size)
        self.c = c
        self.d = d
        self.t = t
        self.stats_samples = stats_samples
        self._name = name
        self.allocation_strategy = allocation_strategy
        self.counter_root: CounterNode
        self.verbose = verbose

        self._path: List[CounterNode] = []

        # A buffer will be used to store the leave before evaluating them by batch of buffer_size
        # _buffer : hash(Leaf) -> CounterNode
        # _buffer_index : hash(Leaf) -> Leaf
        # So that, two leaves with same values will be only evaluating once, even if there are in the same buffer
        self._buffer: Dict[int, List[CounterNode]] = dict()
        self._buffer_idx: Dict[int, Node] = dict()

        if self.verbose:
            print("--- INITIALIZATION ---\n %s\n" % str(self))

    def _search(self, root: Node, nb_of_tree_walks: int) -> Tuple[Node, float]:
        """
        First we accumulate some stats about the tree search in order to be able to correctly parameters
        the ressource allocator then we launch the real search using _single_search method
        This had to be split in order to not recompute several times this step when using meta-search strategy
        """
        stats = TreeStats(root)
        stats.accumulate_stats(nb_samples=self.stats_samples)

        if self.verbose:
            print("A statistic search was performed on the tree\n" "\t- time spent : %0.3fs" % stats.time_spent())
            print(
                "\t- Results :\n"
                "\t\tdepth : %s\n"
                "\t\tbranching factor : %s\n" % (str(stats.depths_info()), str(stats.branching_factors_info()))
            )

        return self._single_search(root, nb_of_tree_walks, stats)

    def _single_search(self, root: Node, nb_of_tree_walks: int, stats: TreeStats):
        """
        Given the root nodes and all the parameters, search for the best leaf
        As describe in section 4.2 of 'Single-Player Monte-Carlo Tree Search for SameGame',
        we allocate ressorce move by move rather than using all the ressources from the tree node.

        Concretely :
        1. Use allocation strategy to determine how many tree walks can be performed from current root
        2. Perform the tree walks and update statistics on the fly
        3. Choose and go to the best children of current root
        4. Repeat until we reach a terminal node

        :param root: Node object from which to perform the tree search
        :param nb_of_tree_walks: total number of tree walks allowed
                -> for now this number is not strictly respected
        """
        nb_of_tree_walks_remaining = nb_of_tree_walks

        # Initialize the resource allocator
        ressource_allocator = RessourceAllocation(
            allocation_strategy=self.allocation_strategy,
            total_ressources=nb_of_tree_walks,
            depth=int(stats.depths_info()["mean"])
        )

        # Wrap the root node in a counter object in order to maintain statistic on the path and rewards
        self.counter_root = CounterNode(reference_node=root, parent=None)

        current_root = self.counter_root

        current_depth = 1
        self._path = [current_root]

        if self.verbose:
            print("--- SEARCHING ---")

        while not current_root.reference_node.is_terminal() and nb_of_tree_walks_remaining > 0:
            nb_of_tree_walks = min(ressource_allocator(current_depth), nb_of_tree_walks_remaining)
            if self.verbose:
                print(
                    "\rCurrent depth %d - will perform %d tree walks" % (current_depth, nb_of_tree_walks), end=" ",
                )
            self._perform_tree_walks(current_root, nb_of_tree_walks)
            nb_of_tree_walks_remaining -= nb_of_tree_walks

            # Freeze current_root to avoid modifying the counter in futur backprops
            # Choose the best node among the childrens and continue the search from here
            current_root.freeze = True
            current_root = current_root.top_children()
            current_depth += 1
            self._path.append(current_root)

        if self.verbose:
            print("\n")

    def _perform_tree_walks(self, current_root: CounterNode, nb_tree_walks: int):
        for _ in range(nb_tree_walks):
            if current_root.solved:
                break

            counter_node, leaf = self._single_tree_walk(current_root)

            # If we already know the value of the leaf,
            # we retrieve it and directly backpropagate it
            if self.heuristic.has_already_eval(leaf):
                leaf_value = self.heuristic.value_from_memory(leaf)
                counter_node.backpropagate(leaf_value, leaf)
            else:  # Store it in the buffer
                self._buffer.setdefault(hash(leaf), []).append(counter_node)
                self._buffer_idx[hash(leaf)] = leaf

            if len(self._buffer) == self.buffer_size:
                self._reduce_buffer()

        if len(self._buffer) > 0:
            self._reduce_buffer()

    def _reduce_buffer(self):
        """
        Reduce the buffer by :
            1/ evaluate all the leaf contained in the buffer
            2/ check if one of the leaf is better than best leaf found so far
            3/ backpropagate the information
            4/ empty the buffer
        """
        leaves = list(self._buffer_idx.values())
        rewards = self.heuristic.eval(leaves)
        for leaf, reward in zip(leaves, rewards):
            for counter_node in self._buffer[hash(leaf)]:
                if reward > self.best_leaf_value:
                    self.best_leaf_value = reward
                    self.best_leaf = leaf
                counter_node.backpropagate(reward, leaf)
            self._buffer = dict()
            self._buffer_idx = dict()

    def _single_tree_walk(self, current_root: CounterNode) -> Tuple[CounterNode, Node]:
        """
        Perform a single tree walk.
        1. The 'bandit phase' / selection step: if current node has been explored so far,
            we choose next node w.r.t selection policy
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
        if not counter_node.reference_node.is_terminal() and counter_node.count > self.t:
            counter_node.expand()

        if counter_node.reference_node.is_terminal():
            counter_node.set_as_solved()  # this also backpropagates the information to previous node if needed

        # perform a random walk from there
        random_leaf = counter_node.reference_node.random_walk()

        return counter_node, random_leaf

    def selection_policy(self, counter_node: CounterNode) -> CounterNode:
        """
        1/ if a children of current node has not been visited yet: visit it
        2/ if all childrens has been visited already once:
            select the children with the biggest UPC value from the node that has not been solved yet
        """
        for children in counter_node.childrens():
            if children.count == 0:
                return children

        total_selection = counter_node.count

        # Select node that maximise UPC + skip node that have been solved
        unsolved_childrens = [children for children in counter_node.childrens() if not children.solved]

        return max(unsolved_childrens, key=lambda node: self.node_upper_confidence_bound(node, total_selection))

    def node_upper_confidence_bound(self, node: CounterNode, total_nb_of_selections: int) -> float:
        """
        Compute the upper confidence bound as describe in section 4.1 of
        'Single-Player Monte-Carlo Tree Search for SameGame'
        """
        return (
            node.sum_rewards / node.count
            + math.sqrt(self.c * math.log(total_nb_of_selections / node.count))
            + math.sqrt(
                (node.sum_of_square_rewards - node.count * ((node.sum_rewards / node.count) ** 2) + self.d)
                / node.count
            )
        )

    def path(self) -> List[Node]:
        assert self._path != [], "Requesting path but no search was launched before"
        return list(map(lambda counter_node: counter_node.reference_node, self._path))

    def counter_path(self) -> List[CounterNode]:
        return self._path  # useful for analyse / debug
