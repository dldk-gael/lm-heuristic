from typing import *
import random
import json

from tree_search.strategy import TreeSearch
from tree_search.tree import Node


class EvalStrategy:
    """
    Class use to evaluate and compare different search strategy on a dataset
    """

    def __init__(
        self, verbose: bool = False,
    ):
        """
        :param verbose: use True to print information about evaluation progress
        """
        self.eval_results = dict()
        self.verbose = verbose

    def __call__(
        self,
        strategies: Union[TreeSearch, List[TreeSearch]],
        dataset: Union[Node, List[Node]],
        nb_random_restarts: int = 1,
        nb_of_tree_walks: int = 1,
    ):
        """
        :param strategies: list (or single) search strategy to evaluate
        :param dataset: list of tree that will be used to evaluate the search strategies
        :param nb_random_restarts: number of search to perform on each example of the dataset
                                    a different random seed will be used at each restart
        :param nb_of_tree_walks: number of tree_walks
        """
        self.eval_results = dict()

        strategies = [strategies] if isinstance(strategies, TreeSearch) else strategies
        dataset = [dataset] if isinstance(dataset, Node) else dataset

        for strategy in strategies:
            strategy_name = str(strategy)
            self.eval_results[strategy_name] = dict()
            if self.verbose:
                print("Evaluating %s ...", strategy_name)
            for i, root_sample in enumerate(dataset):
                self.eval_results[strategy_name][i] = []
                for j in range(nb_random_restarts):
                    random.seed(j)
                    strategy.search(root_sample, nb_of_tree_walks=nb_of_tree_walks)
                    search_info = strategy.search_info()

                    self.eval_results[strategy_name][i].append(
                        {k: search_info[k] for k in ("time", "best_leaf_value")}
                    )

    def save_results(self, path):
        with open(path, "w") as file:
            json.dump(self.eval_results, file)

    @classmethod
    def from_json(cls, path):
        eval_strategy = cls(verbose=False)
        with open(path, "r") as file:
            eval_strategy.eval_results = json.load(file)
        return eval_strategy
