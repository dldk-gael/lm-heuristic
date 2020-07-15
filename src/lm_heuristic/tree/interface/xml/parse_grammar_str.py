from functools import reduce
import re
from .feature_structure import PConstant, PVar, PStruct, copy_features, revert

dead_ends = 0  # Just hacked in this dead end thing to check some stuff

def generate_sentences(grammar, accumulator, sentences):
    global dead_ends
    all_terminal = True
    new_accumulator = []
    for i in range(len(accumulator)):
        symbol = accumulator[i]
        if symbol["str"].startswith('"') or symbol["str"].startswith("'"):
            new_accumulator.append(symbol)
        else:
            all_terminal = False
            bodies = grammar.get(symbol["str"], [])
            if not bodies:
                continue
            selection = bodies  # random.sample(bodies, min(len(bodies), 1))
            non_matches = 0
            for body in selection:
                # make new variables for body
                feature_copies = copy_features(body)  # copy features  of type {'str': 'S', 'features': [...]}
                head_bindings = []
                if not symbol["features"].unify(feature_copies[0]["features"], head_bindings):
                    revert(head_bindings)
                    non_matches += 1
                    continue
                # print(new_accumulator + feature_copies[1:] + accumulator[i+1:])
                generate_sentences(grammar, new_accumulator + feature_copies[1:] + accumulator[i + 1 :], sentences)
                revert(head_bindings)
            if non_matches == len(bodies):
                print("feature dead end:", [a["str"] + str(a["features"].show()) for a in accumulator])
                dead_ends += 1  # <---- dead end due to no possible submatches given the feature bindings that were made earlier. The more 'right' in the grammar exploration tree this happens the more computational cost this incurs
            return
    if all_terminal:
        sentences.append(" ".join(token["str"][1:-1] for token in accumulator))
    else:
        print("dead end due to unknown symbol: ", [a["str"] + str(a["features"].show()) for a in accumulator])
        dead_ends += 1  # <---- dead symbols, basically a grammar error



def parse_grammar(as_str):
    lines = as_str.splitlines()
    grammar = {}
    for line in lines:
        line_split = line.split("->")
        if len(line_split) == 1 or line_split[0].startswith("#"):
            continue
        head = "".join(line_split[0].split())
        if "[" in head:
            head = head[: head.index("[")]
        grammar[head] = grammar.get(head, [])
        for body_split in line_split[1].split("|"):
            symbols, features = parse_rule("".join(line_split[0].split()) + "->" + body_split)
            grammar[head].append(
                {"head_feature": features[0], "body_symbols": symbols[1:], "body_features": features[1:]}
            )
    return grammar


def parse_rule(str):
    variables = {}
    for name in re.findall(r"\?[A-Za-z0-9]+", str):
        variables[name] = PVar(name)
    str = str.strip()
    rule_split = str.split("->")
    head, head_feature = get_feature_struct(rule_split[0].strip(), variables)
    symbols = [head]
    features = [head_feature]
    for b in rule_split[1].strip().split(r" "):
        if not b:
            continue
        symbol, feature = get_feature_struct(b, variables)
        symbols.append(symbol)
        features.append(feature)
    return symbols, features


def parse_separate_feature(str, old_variables={}):
    variables = {}
    for name in re.findall(r"\?[A-Za-z0-9]+", str):
        if name in old_variables:
            variables[name] = old_variables[name]
        else:
            variables[name] = PVar(name)
    _, feature = get_feature_struct(str.strip(), variables)
    return feature, variables


def get_feature_struct(str, variables):
    if "[" not in str:
        return str, PStruct({})
    str2 = str[str.index("[") + 1 : -1]
    arguments = []
    record = ""
    recording = True
    nested = 0
    for i in str2:
        if recording and nested == 0 and i == ",":
            recording = False
            arguments.append(record)
            record = ""
        if i == "[":
            nested += 1
        if i == "]":
            nested -= 1
        if not recording and (i != "," and i != " "):
            recording = True
        if recording:
            record += i
    arguments.append(record)
    features = {}
    for a in arguments:
        if "=" not in a:
            print(str)
        s = a.index("=")
        name = a[:s]
        value = a[s + 1 :]
        if value in variables:
            features[name] = variables[value]
        elif value[0] == "[":
            _, features[name] = get_feature_struct(value, variables)
        else:
            features[name] = PConstant(value)
    return str[: str.index("[")], PStruct(features)


def get_combination(options, i):
    combination = []
    nr_combinations = reduce(lambda x, y: x * y, [len(z) for z in options])
    for entry in options:
        combination.append(entry[int((i / nr_combinations) * len(entry))])
        nr_combinations = nr_combinations / len(entry)
        i = i % nr_combinations
    return combination


