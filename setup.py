from setuptools import setup

setup(
    name="lm-heuristic",
    version="",
    packages=[
        "lm_heuristic",
        "lm_heuristic.cfg",
        "lm_heuristic.tree",
        "lm_heuristic.utils",
        "lm_heuristic.heuristic",
        "lm_heuristic.heuristic.sentence_score",
        "lm_heuristic.evaluation",
        "lm_heuristic.tree_search",
        "lm_heuristic.tree_search.mcts",
        "lm_heuristic.tree_search.random",
    ],
    install_requires=[
        "matplotlib",
        "torch",
        "pandas",
        "seaborn",
        "nltk",
        "numpy",
        "transformers",
        "tqdm",
        "lm-scorer"
    ],
    url="https://github.com/dldk-gael/lm-heuristic",
    license="",
    author="Gaël de Léséleuc",
    author_email="gael@de-leseleuc.com",
    description="",
)
