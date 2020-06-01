/**
* This module defined several predicates to work with grammar in Prolog
* The grammar must be written using two predicates : 
* - terminal(X)  
*     -> assert that X is a terminal symbol
*
* - rule(LHS, RHS) 
*     -> assert that RHS (a list of term) can be derivated from LHS (a unique term)
*/


/**
* child :
*   - arg0 : a current derivation, eg: ["gael", v, np(obj)]
*   - arg1 : new derivation that derivated from arg0 using only one rules, 
*        eg: ["gael", "like", np(ojb)] or ["gael", v, det(ojb), n(obj)] 
*/
child([], []).
child([Symb|Q], [Symb|X]) :- child(Q, X), Q \= X.
child([Symb|Q], X) :- rule(Symb, Rhs), append(Rhs, Q, X).


/**
* all_children :
* - arg0 : a current derivation, eg: ["gael", v, np(obj)]
* - arg1 : all derivations that can be derivated from arg0 using only one rules 
*        -> List[List[Symbol]]
*/
all_children(Derivation, Children) :- findall(X, child(Derivation, X), Children).


/** 
* leaf : 
*   - arg0 : a current derivation, eg: ["gael", v, np(obj)]
*   - arg1 : a derivation composed only of terminal symbols that derivated from arg0,
*        eg: ["gael", "like", "football"]
*   Note that if arg0 is already a leaf, arg1 will be set to arg0 
*/
leaf([], []).
leaf([Symb|Q], TerminalDerivation) :- 
    (
        terminal(Symb) -> leaf(Q, Z), TerminalDerivation = [Symb|Z];
        rule(Symb, Symb_Rhs), leaf(Symb_Rhs, Y), leaf(Q, Z), append(Y, Z, TerminalDerivation)
    ).

/**
* all_valid_children :
* - arg0 : a current derivation, eg: ["gael", v, np(obj)]
* - arg1 : all derivations that can be derivated from arg0 using only one rules 
*           AND for each of these derivation, a leaf can be reached 
*           (this is intended to be used for FeatureGrammar)
*/
filter_valid_children([], []).
filter_valid_children([Child|Q], ValidChildren) :-
    filter_valid_children(Q, Vq),
    % Have to make a copy in order not to return generic Variable in the results
    copy_term(Child, ChildCopy), 
    (leaf(ChildCopy, _) -> append([Child], Vq, ValidChildren);
    ValidChildren = Vq).

all_valid_children(Derivation, ValidChildren) :-
    all_children(Derivation, Children), 
    filter_valid_children(Children, ValidChildren).


/**  
* random_leaf : same as leaf but create the solution in a random order
*/
rules(LHS, Rules) :- findall(rule(LHS,RHS), rule(LHS,RHS), Rules).
random_rules(LHS, Rules) :- rules(LHS, X), random_permutation(X, Rules).
random_rule(LHS, RHS) :- random_rules(LHS, Rules), member(rule(LHS, RHS), Rules).

random_leaf([], []).
random_leaf([Symb|Q], TerminalDerivation) :- 
    (
        terminal(Symb) -> random_leaf(Q, Z), TerminalDerivation = [Symb|Z];
        random_rule(Symb, Symb_Rhs), random_leaf(Symb_Rhs, Y), random_leaf(Q, Z), append(Y, Z, TerminalDerivation)
    ).