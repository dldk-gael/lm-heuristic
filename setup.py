from setuptools import setup, find_packages

setup(
    name="lm-heuristic",
    version="0.1.0",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "matplotlib",
        "torch",
        "pandas",
        "seaborn",
        "nltk",
        "numpy",
        "transformers",
        "tensorflow",
        "tensorflow_hub",
        "flask",
        "celery",
        "pyswip"
    ],
    url="https://github.com/dldk-gael/lm-heuristic",
    author="Gaël de Léséleuc",
    author_email="gael@de-leseleuc.com",
    description="Package to found the most natural sentence that can be generated by a given context-free grammar",
)
