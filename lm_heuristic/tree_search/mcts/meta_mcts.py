import random

from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch
from lm_heuristic.tree import Node, TreeStats


class RandomRestartsMCTS(MonteCarloTreeSearch):
    """
    This class implements the meta-heuristic proposed in
    Single-Player Monte-Carlo Tree Search for SameGame by Schadda, Winandsan, Taka, Uiterwijka

    The idea is that, for the same computationnal ressources, it can be better to randomly restart
    the MCTS several times and keep the best value rather that using all the computationnal ressources
    in one single MCTS
    """

    def __init__(self, nb_random_restarts: int = 1, **mcts_parameters):
        MonteCarloTreeSearch.__init__(self, **mcts_parameters)
        self.nb_random_restarts = nb_random_restarts

    def _search(self, root: Node, nb_of_tree_walks: int) -> (Node, float):
        """
        As in MCTS, accumulate some stats about the tree.
        But now, it will divide the nb_of_tree_walks between the nb_random_restarts search and keep the
        best results
        """
        stats = TreeStats(root)
        stats.accumulate_stats(nb_samples=100)

        if self.verbose:
            print("A statistic search was performed on the tree\n" "\t- time spent : %0.3fs" % stats.time_spent())
            print(
                "\t- Results :\n"
                "\t\tdepth : %s\n"
                "\t\tbranching factor : %s\n" % (str(stats.depths_info()), str(stats.branching_factors_info()))
            )

        tree_walks_per_search = nb_of_tree_walks // self.nb_random_restarts
        best_leaf, best_value, best_path = None, 0, []

        for i in range(self.nb_random_restarts):
            if self.verbose:
                print("Launching restart n°", i)
            # for last search, we add the rest of computationnal ressources
            if i == self.nb_random_restarts - 1:
                tree_walks_per_search += nb_of_tree_walks % self.nb_random_restarts

            random.seed()  # this is probably useless
            leaf, value = self._single_search(root, tree_walks_per_search, stats)

            if value > best_value:
                best_leaf, best_value, best_path = leaf, value, self._path
            self._path = []
        self._path = best_path

        return best_leaf, best_value
