# lm-heuristic

## Project description

The purpose of this package is to find the most *natural* sentence among all sentences 
that can be been generated by a given context free grammar. 

## Setup

```
git clone https://github.com/dldk-gael/lm-heuristic.git
cd lm-heuristic
pip install .
```

## Package architecture 

- **sentence_score** : interface towards transformers-based model (GPT2 and BERT) 
that are used to associate a sentence with a *naturalness* score. This module will be replace by [lm-scorer](https://github.com/simonepri/lm-scorer) once this library will support input batching. 

- **tree** : define 
    - an abstract class **Node** from which all tree structure must inheritate 
    - **CFGrammarNode** which is constructed from a NLTK CFG
    - **CounterNode** which is a wrapper use to keep statistics on other nodes a 
    - **TreeStats** which can be used to accumulate statistics on a given tree. 

- **heuristic** : use to wrap an evaluation function (that takes as input a list of **Node** and return the associated list of scores). Moreover it adds on top of the evaluation function a memory buffer and keep also statistics about the call to the evaluation function. 

- **tree_search** : define an abstract **TreeSearch** class which is a search strategy that must be initialized with an **Heuristic**. Then given a root (**Node**) and a number of allowed walks, it return the best leaf that has been found when. For now, two tree search strategy are implemented : 
     1. **RamdomSearch** that sample randomly different path 
     2. **MonteCarloTreeSearch** that keep statistics on the best path discover so far using **CounterNode** 
  
- **evaluation** : define an experimentation framework **EvalStrategy**. It take as input : a list of tree search strategy, 
a tree dataset, perform different type of evaluation on it and store the results all of the experiments in a panda dataframe.
 
## Examples

Various examples on how to use all those module can be found in *examples/*
