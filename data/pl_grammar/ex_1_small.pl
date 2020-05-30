rule(s, [arg_0, v, arg_1]).
rule(v, [v_want]).
rule(arg_0, [np_arg_0]).
rule(np_arg_0, [det, noun_boy]).
rule(np_arg_0, [noun_boy]).
rule(np_arg_0, [subj_male]).
rule(arg_1, [pp_arg_1]).
rule(pp_arg_1, ['to', vp_arg_1]).
rule(pp_arg_1, [vp_arg_1]).
rule(vp_arg_1, [v_ride, np_arg1_arg0]).
rule(np_arg1_arg0, [det, adj_mod, noun_bycicle]).
rule(np_arg1_arg0, [det, noun_bycicle, rel_clause_mod]).
rule(rel_clause_mod, [rel, v_be, adj_mod]).
rule(noun_boy, ['boy']).
rule(noun_bycicle, ['bicycle']).
rule(v_want, ['want']).
rule(v_want, ['wanting']).
rule(v_want, ['wants']).
rule(v_be, ['is']).
rule(v_be, ['being']).
rule(v_be, ['be']).
rule(v_be, ['are']).
rule(v_be, ['was']).
rule(det, ['a']).
rule(det, ['the']).
rule(det, ['one']).
rule(det, ['this']).
rule(det, ['those']).
rule(det, ['an']).
rule(subj_male, ['he']).
rule(subj_male, ['him']).
rule(subj_male, ['his']).
rule(subj_neutral, ['it']).
rule(subj_neutral, ['that']).
rule(rel, ['that']).
rule(rel, ['which']).
rule(v_ride, ['ride']).
rule(v_ride, ['riding']).
rule(v_ride, ['rides']).
rule(adj_mod, ['red']).
terminal('a').
terminal('ride').
terminal('is').
terminal('wanting').
terminal('this').
terminal('those').
terminal('was').
terminal('which').
terminal('an').
terminal('to').
terminal('being').
terminal('boy').
terminal('want').
terminal('rides').
terminal('one').
terminal('his').
terminal('it').
terminal('be').
terminal('that').
terminal('red').
terminal('bicycle').
terminal('him').
terminal('riding').
terminal('the').
terminal('are').
terminal('wants').