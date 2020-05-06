from typing import *
import random
import pandas as pd

from tree_search import TreeSearch
from tree import Node


class EvalStrategy:
    """
    Class use to evaluate and compare different search strategy on a dataset
    The idea is to evaluate each strategy :
    - on different tree
    - with different number of allowed tree walks
    - with different random seeds

    This class is use to run all the experiments and produce a dataframe
    gathering all the results
    """

    def __init__(
        self, verbose: bool = False,
    ):
        """
        :param verbose: use True to print information about evaluation progress
        """
        self.eval_results = pd.DataFrame(
            columns=[
                "strategy",
                "dataset",
                "nb_tree_walks",
                "random_seed",
                "time_needed",
                "best_value",
            ]
        )
        self.verbose = verbose

    def __call__(
        self,
        strategies: Union[TreeSearch, List[TreeSearch]],
        dataset: Union[Node, List[Node]],
        nb_tree_walks: Union[int, List[int]],
        nb_random_restarts: int,
    ) -> pd.DataFrame:
        """
        :param strategies: list (or single) search strategy to evaluate
        :param dataset: list of tree that will be used to evaluate the search strategies
        :param nb_tree_walks: number of tree_walks to try for each strategy
        :param nb_random_restarts: number of search to perform on each example of the dataset
                                    a different random seed will be used at each restart
        """
        experiments_list = []

        strategies = [strategies] if isinstance(strategies, TreeSearch) else strategies
        dataset = [dataset] if isinstance(dataset, Node) else dataset
        nb_tree_walks = (
            [nb_tree_walks] if isinstance(nb_tree_walks, int) else nb_tree_walks
        )

        for strategy in strategies:
            strategy_name = str(strategy)
            if self.verbose:
                print("\rEvaluating %s ..." % strategy_name)
            for i, root_sample in enumerate(dataset):
                for k in nb_tree_walks:
                    for j in range(nb_random_restarts):
                        if self.verbose:
                            print("\rCurrent evaluation : example n°%d "
                                  "with %d tree walks and random_seed(%d)" % (i, k, j), end="")
                        random.seed(j)
                        best_leaf, best_leaf_value = strategy(root_sample, nb_of_tree_walks=k)
                        time_needed = strategy.time_spent()

                        experiment_results = {
                            "strategy": strategy_name,
                            "dataset": i,
                            "nb_tree_walks": k,
                            "random_seed": j,
                            "time_needed": time_needed,
                            "best_value": best_leaf_value,
                            "best_leaf": str(best_leaf)
                        }
                        experiments_list.append(experiment_results)
        if self.verbose:
            print("\n")
        self.eval_results = pd.DataFrame(experiments_list)
        return self.eval_results
