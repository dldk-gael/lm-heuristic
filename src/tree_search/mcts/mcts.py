import math
from typing import *

from heuristic import Heuristic
from tree_search import TreeSearch
from .allocation_strategy import AllocationStrategy, RessourceAllocation
from tree import CounterNode, Node, TreeStats


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
        allocation_strategy: AllocationStrategy = AllocationStrategy.UNIFORM,
        verbose=False,
    ):
        """
        Initialize the MCTS paramater and create the counter object that will be use to perform the MCTS
        :param heuristic: Heuristic instance
        :param buffer_size: number of leaf to store in memory before evaluating them in one pass
        :param c: hyperparameter for upper confidence bound, control the exploration vs exploitation ratio
        :param d: hyperparameter for upper confidence bound
        :param t: threshold for expansion policy (see expansion_policy method)
        :param allocation_strategy: strategy that determine how many tree walks will be performed at each layer
        """
        TreeSearch.__init__(self, heuristic, buffer_size)
        self.c = c
        self.d = d
        self.t = t
        self.allocation_strategy = allocation_strategy
        self._path = []
        self.counter_root = None
        self.verbose = verbose

        if self.verbose:
            print("--- INITIALIZATION ---\n %s\n" % str(self))

    def _search(self, root: Node, nb_of_tree_walks: int) -> (Node, float):
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
        assert (
            nb_of_tree_walks > self.t
        ), "You give a lower number of tree walks that threshold t : the root node will never expand"

        # Perfom few quick walks to assess tree's depth (time is negligeable compare to rest of algorithms)
        stats = TreeStats(root)
        stats.accumulate_stats(nb_samples=100)
        mean_depth = int(stats.depths_info()["mean"])
        if self.verbose:
            print(
                "A statistic search was performed on the tree\n"
                "\t- time spent : %0.3fs" % stats.time_spent()
            )
            print(
                "\t- Results :\n"
                "\t\tdepth : %s\n"
                "\t\tbranching factor : %s\n"
                % (str(stats.depths_info()), str(stats.branching_factors_info()))
            )

        # Initialize the resource allocator
        resource_allocator = RessourceAllocation(
            allocation_strategy=self.allocation_strategy,
            total_ressources=nb_of_tree_walks,
            max_depth=mean_depth,  # this could end up with computing more tree walks than allowed
            min_ressources_per_move=10,  # this has been fixed arbitrarly for now
        )

        # Wrap the root node in a counter object in order to maintain statistic on the path and rewards
        self.counter_root = CounterNode(reference_node=root, parent=None)

        current_root = self.counter_root

        current_depth = 1
        self._path = [current_root]

        if self.verbose:
            print("--- SEARCHING ---")

        while not current_root.reference_node.is_terminal():
            nb_of_tree_walks = resource_allocator(current_depth)
            if self.verbose:
                print(
                    "\rCurrent depth %d - will perform %d tree walks"
                    % (current_depth, nb_of_tree_walks),
                    end=" ",
                )
            # Perform nb of tree walks
            self._perform_tree_walks(current_root, nb_of_tree_walks)

            # Freeze current_root to avoid modifying the counter in futur backprops
            # Choose the best node among the childrens and continue the search from here
            current_root.freeze = True
            current_root = current_root.top_children()
            current_depth += 1
            self._path.append(current_root)

        if self.verbose:
            print("\n")

        return (
            current_root.reference_node,
            current_root.top_reward,
        )

    def _perform_tree_walks(self, current_root: CounterNode, nb_tree_walks: int):
        # use a buffer to store in memory couple(counter_node, leaf)
        # before evaluating them in one single pass when the buffer is full
        # this is usefull for evaluation function that can beneficiate of //
        # by using a dict and not a list for the buffer representation,
        # we can handle the case where same leaf end up several times in the buffer
        buffer = dict()  # dict (hash(leaf)  -> List[counter_node])
        buffer_idx = dict()  # dict (hash(leaf) -> leaf)

        def flush_buffer():
            """
            1/ evaluate all the leaf contained in the buffer
            2/ backpropagate the information
            3/ empty the buffer
            """
            nonlocal buffer, buffer_idx, self
            leafs = list(buffer_idx.values())
            rewards = self.heuristic.eval(leafs)
            for _leaf, _reward in zip(leafs, rewards):
                for _counter_node in buffer[hash(_leaf)]:
                    _counter_node.backpropagate(_reward, _leaf)
            buffer = dict()
            buffer_idx = dict()

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
                buffer.setdefault(hash(leaf), []).append(counter_node)
                buffer_idx[hash(leaf)] = leaf

            if len(buffer) == self.buffer_size:
                flush_buffer()

        if len(buffer) > 0:
            flush_buffer()

    def _single_tree_walk(self, current_root: CounterNode) -> (CounterNode, Node):
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
        if (
            not counter_node.reference_node.is_terminal()
            and counter_node.count + 1 > self.t
        ):
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
        unsolved_childrens = [
            children for children in counter_node.childrens() if not children.solved
        ]
        assert len(unsolved_childrens) > 0, "Try to select from a solved node"

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
        assert self._path != [], "Requesting path but no search was launched before"
        return list(map(lambda counter_node: counter_node.reference_node, self._path))

    def counter_path(self) -> List[CounterNode]:
        return self._path  # useful for analyse / debug

    def __str__(self):
        return "MCTS c=%d d=%d t=%d" % (self.c, self.d, self.t)
