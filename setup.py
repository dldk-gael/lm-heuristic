from setuptools import setup, find_packages

setup(
    name='lm_heuristic',
    version='0.1-dev',
    packages=find_packages("lm_heuristic"),
    package_dir={'': 'lm_heuristic'},
    install_requires=[
        "matplotlib",
        "torch",
        "pandas",
        "seaborn",
        "nltk",
        "numpy",
        "transformers",
        "tqdm"
    ],
    url='https://github.com/dldk-gael/lm-heuristic',
    license='',
    author='Gaël de Léséleuc de Kérouara',
    author_email='gael@de-leseleuc.com',
    description=''
)
