from typing import *
import time

import pandas as pd

from lm_heuristic.tree_search import TreeSearch
from lm_heuristic.tree import Node


class Benchmark:
    """
    Class use to evaluate and compare different search strategy on a dataset
    The idea is to evaluate each strategy :
    - on different tree
    - with different number of allowed tree walks
    - with different random seeds

    This class is use to run all the experiments and produce a dataframe that gather all the results
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
                "input_nb_tree_walks",
                "nb_of_unique_leaf_eval",
                "restart",
                "eval_call_counter",
                "best_value",
            ]
        )
        self.verbose = verbose

    def __call__(
        self,
        strategies: Union[TreeSearch, List[TreeSearch]],
        dataset: Union[Tuple[Node, str], List[Tuple[Node, str]]],
        nb_tree_walks: Union[int, List[int]],
        nb_random_restarts: int,
    ) -> pd.DataFrame:
        """
        :param strategies: list (or single) search strategy to evaluate
        :param dataset: list of tree and their associated name that will be used to evaluate the search strategies
        :param nb_tree_walks: number of tree_walks to try for each strategy
        :param nb_random_restarts: number of search to perform on each example of the dataset
                                    a different random seed will be used at each restart
        """
        experiments_list = []

        strategies = [strategies] if isinstance(strategies, TreeSearch) else strategies
        dataset = dataset if isinstance(dataset, list) else [dataset]
        nb_tree_walks = [nb_tree_walks] if isinstance(nb_tree_walks, int) else nb_tree_walks

        for strategy in strategies:
            strategy_name = str(strategy)
            for root_sample, sample_name in dataset:
                for k in nb_tree_walks:
                    for j in range(nb_random_restarts):
                        if self.verbose:
                            print(
                                "\rCurrent evaluation : %s - example %s "
                                "- %d tree walks "
                                "- random restart n°%d" % (strategy_name, sample_name, k, j),
                                end="",
                            )
                        begin_time = time.process_time()
                        best_leaf, best_leaf_value = strategy.search(root_sample, nb_of_tree_walks=k)
                        time_needed = time.process_time() - begin_time

                        experiment_results = {
                            "strategy": strategy_name,
                            "dataset": sample_name,
                            "input_nb_tree_walks": k,
                            "restart": j,
                            "time_needed": time_needed,
                            "best_value": best_leaf_value,
                            "best_leaf": str(best_leaf),
                        }
                        experiments_list.append(experiment_results)
        if self.verbose:
            print("\n")
        self.eval_results = pd.DataFrame(experiments_list)
        return self.eval_results
