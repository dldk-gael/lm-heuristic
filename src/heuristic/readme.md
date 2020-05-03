I am currently contributing to lm-scorer library so that it support input batching.
As soon as this feature will be supported by lm-score, this module will be removed and I will directly used
lm-scorer library in the code.

There is also some minor differences between this code and lm-score library, for instance:
I do not add EOS special token at the end of sentences, but this should not have an impact on the ranking