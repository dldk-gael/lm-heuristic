from functools import reduce 
from timeit import timeit


def test_grammar():
    as_str = """ 
        s -> a b
        a -> 'A'
        b -> 'B'
        """
        
    grammar = parse_grammar(as_str)
    print("Test generate all")
    sentences = []
    generate_all(grammar, ['s'], sentences)
    for s in sentences:
        print(': ' + s + '.')

    print("Traverse grammar")
    traverse_grammar(grammar, ['s'])

    
def generate_all(grammar, accumulator, sentences): 
    all_terminal = True
    new_accumulator = []
    for i in range(len(accumulator)):
        symbol = accumulator[i]
        if symbol.startswith("'"):
            new_accumulator.append(accumulator[i])
        else:
            all_terminal = False
            bodies = grammar[symbol]
            for body in bodies:
                generate_all(grammar, new_accumulator + body + accumulator[i+1:], sentences)
            break
    if all_terminal:
        sentences.append(" ".join(token[1:-1] for token in accumulator))


def traverse_grammar(grammar, symbols): 
    choices = []
    all_terminal = True
    for symbol in symbols:
        if symbol.startswith("'"):
            choices.append([[symbol]])
        else:
            choices.append(grammar[symbol])
            all_terminal = False
    nr_children = reduce(lambda x,y:x*y, [len(c) for c in choices]) 
    if all_terminal:
        print(" ".join([c[0][0][1:-1] for c in choices]))
        return 
    else:
        for child in range(nr_children):
            new_symbols = get_combination(choices, child)
            traverse_grammar(grammar, reduce(lambda x,y: x+y, new_symbols))
    
def get_next_child(tree):
    if tree['symbol'].startswith("'"):
        return {}
    elif tree['done']:
        return {}
    elif not tree['children']: 
        return tree
    else:
        for child in tree['children']:
            next = get_next_child(child)
            if next:
                return next
        tree['done'] = True
        return {}

def parse_grammar(as_str):
    lines = as_str.splitlines()
    grammar = {}
    for line in lines:
        line_split = line.split('->')
        if len(line_split) == 1:
            continue
        head = "".join(line_split[0].split())
        grammar[head] = grammar.get(head, [])
        for body_split in line_split[1].split('|'):
            body = body_split.split()       
            grammar[head].append(body)
    return grammar
    
def get_combination(options, i):
    combination = []
    nr_combinations = reduce(lambda x, y: x * y,[len(z) for z in options])
    for entry in options: 
        combination.append(entry[int((i/nr_combinations) * len(entry))])
        nr_combinations = nr_combinations / len(entry) 
        i = i % nr_combinations
    return combination


if __name__ == '__main__':
    test_grammar()
