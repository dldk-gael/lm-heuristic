/**
* Define two predicates to work with grammar 
* child :
*   - arg0 : a current derivation, eg: ["gael", v, np(obj)]
*   - arg1 : new derivation that derivated from arg0 using only one rules, 
*        eg: ["gael", "like", np(ojb)] or ["gael", v, det(ojb), n(obj)] 
*
* leaf : 
*   - arg0 : a current derivation, eg: ["gael", v, np(obj)]
*   - arg1 : a derivation composed only of terminal symbols that derivated from arg0,
*        eg: ["gael", "like", "football"]
*/

child([], []).
child([Symb|Q], [Symb|X]) :- child(Q, X), Q \= X.
child([Symb|Q], X) :- rule(Symb, Rhs), append(Rhs, Q, X).

leaf_from([], []).
leaf_from([Symb|Q], X) :- 
    (
        terminal(Symb) -> leaf_from(Q, Z), X = [Symb|Z];
        rule(Symb, Symb_Rhs), leaf_from(Symb_Rhs, Y), leaf_from(Q, Z), append(Y, Z, X)
    ).

leaf(Derivation, X) :- leaf_from(Derivation, X), X \= Derivation.