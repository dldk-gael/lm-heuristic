import xml.etree.ElementTree as ET

def xml_to_string(file):
    grammar_classes, terminals, semantic_structures, nonterminals = parse(file)
    as_str = ""
    for n in [x for x in nonterminals if nonterminals[x]["previous"] == None]:
        nonterminal = nonterminals[n]
        s = nonterminal["symbol"] + feature_struct_to_str(nonterminal) + " -> "
        next = nonterminal["next"]
        while next != None:
            nonterminal = nonterminals[next]
            s += nonterminal["symbol"] + feature_struct_to_str(nonterminal) + " "
            next = nonterminal["next"]
        as_str += s + "\r\n"

    as_str = (
        """  
        """
        + as_str
    )
    for t in [terminals[x] for x in terminals]:
        features_str = ",".join(t["features"])
        as_str += t["class"] + "[" + features_str + "]" + " -> " + "'" + t["symbol"] + "'\r\n"
    
    return as_str

def parse(file):
    dictionaries = {}
    root = ET.parse(file).getroot()

    # grammar classes
    grammar_classes = {}
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "style" in attributes:
            continue
        style = attributes["style"]
        if "rounded=1" in style and "verticalAlign=top" in style and not "text" in style:
            grammar_classes[attributes["value"]] = grammar_classes.get(attributes["value"], list())
            grammar_classes[attributes["value"]].append(
                {
                    "x1": int(float(child[0].attrib["x"])),
                    "y1": int(float(child[0].attrib["y"])),
                    "x2": int(float(child[0].attrib["x"])) + int(float(child[0].attrib["width"])),
                    "y2": int(float(child[0].attrib["y"])) + int(float(child[0].attrib["height"])),
                }
            )

    # terminals
    terminals = {}
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "style" in attributes:
            continue
        style = attributes["style"]
        if "rounded=1" in style and "collapsible=1" in style:
            dictionary = {"id": attributes["id"], "name": attributes["value"], "type": "terminal"}
            dictionaries[dictionary["id"]] = dictionary
            grammar_class = "unknown"
            x = int(float(child[0].attrib["x"]))
            y = int(float(child[0].attrib["y"]))
            for c in grammar_classes:
                for box in grammar_classes[c]:
                    if box["x1"] < x and box["x2"] > x and box["y1"] < y and box["y2"] > y:
                        grammar_class = c
                        break
                if grammar_class != "unknown":
                    break
            terminals[dictionary["id"]] = {"features": [], "class": grammar_class, "symbol": dictionary["name"]}

    # terminal features
    for child in root[0][0][0]:
        attributes = child.attrib
        if "parent" not in attributes or attributes["parent"] not in dictionaries:
            continue
        parent = dictionaries[attributes["parent"]]
        if parent["type"] == "terminal":
            terminals[parent["id"]]["features"].append(attributes["value"])

    # semantic structures
    semantic_structures = {}
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "style" in attributes:
            continue
        style = attributes["style"]
        if "dashed=1" in style and "collapsible=1" in style:
            dictionary = {"id": attributes["id"], "name": attributes["value"], "type": "semantic_structure"}
            dictionaries[dictionary["id"]] = dictionary
            semantic_structures[dictionary["name"]] = {"features": {}}

    # semantic structure features
    for child in root[0][0][0]:
        attributes = child.attrib
        if "parent" not in attributes or attributes["parent"] not in dictionaries:
            continue
        parent = dictionaries[attributes["parent"]]
        dictionaries[attributes["id"]] = {
            "id": attributes["id"],
            "type": "semantic_structure_feature",
            "name": attributes["value"],
            "parent": attributes["parent"],
        }
        if parent["type"] == "semantic_structure":
            semantic_structures[parent["name"]]["features"][attributes["value"]] = {"type": "var"}

    # semantic structure substructure relation
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "edge" in attributes:
            continue
        source = attributes["source"]
        target = attributes["target"]
        if not source in dictionaries or not target in dictionaries:
            continue
        source = dictionaries[source]
        target = dictionaries[target]
        if not source["type"] == "semantic_structure_feature" or not target["type"] == "semantic_structure":
            continue
        parent = semantic_structures[dictionaries[source["parent"]]["name"]]
        target = target["name"]
        parent["features"][source["name"]] = {"type": "structure", "reference": target}

    # semantic structure extension
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "edge" in attributes:
            continue
        source = attributes["source"]
        target = attributes["target"]
        if not source in dictionaries or not target in dictionaries:
            continue
        source = dictionaries[source]
        target = dictionaries[target]
        if not source["type"] == "semantic_structure" or not target["type"] == "semantic_structure":
            continue
        semantic_structures[source["name"]]["features"].update(semantic_structures[target["name"]]["features"])

    # Non-terminals
    nonterminals = {}
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "style" in attributes:
            continue
        style = attributes["style"]
        if not "rounded=1" in style and "collapsible=1" in style and not "dashed=1" in style:
            dictionary = {"id": attributes["id"], "name": attributes["value"], "type": "nonterminal"}
            dictionaries[dictionary["id"]] = dictionary

            nonterminals[dictionary["id"]] = {
                "id": dictionary["id"],
                "features": {},
                "symbol": dictionary["name"],
                "next": None,
                "previous": None,
            }
    # Rules as linked lists
    for child in root[0][0][0]:
        attributes = child.attrib
        if not "edge" in attributes:
            continue
        source = attributes["source"]
        target = attributes["target"]
        if not source in dictionaries or not target in dictionaries:
            continue
        source = dictionaries[source]
        target = dictionaries[target]
        if not source["type"] == "nonterminal" or not target["type"] == "nonterminal":
            continue
        nonterminals[source["id"]]["next"] = target["id"]
        nonterminals[target["id"]]["previous"] = source["id"]

    # Nonterminal feature structures
    for nonterminal in [nonterminals[t] for t in nonterminals if nonterminals[t]["previous"] == None]:
        struct_vars = {}
        get_non_terminal_features(nonterminal, root[0][0][0], dictionaries, semantic_structures, struct_vars)
        next = nonterminal["next"]
        while next != None:
            next_nonterminal = nonterminals[next]
            get_non_terminal_features(next_nonterminal, root[0][0][0], dictionaries, semantic_structures, struct_vars)
            next = next_nonterminal["next"]
    return grammar_classes, terminals, semantic_structures, nonterminals


def get_non_terminal_features(nonterminal, xmlentries, dictionaries, semantic_structures, struct_vars):
    for child in xmlentries:
        attributes = child.attrib
        if "parent" not in attributes or attributes["parent"] != nonterminal["id"]:
            continue
        base_structure = has_base_structure(xmlentries, attributes["id"], dictionaries, semantic_structures)
        name, feature = translate_feature(attributes["value"], semantic_structures, struct_vars, base_structure)
        nonterminal["features"][name] = feature


def translate_feature(as_string, semantic_structures, struct_vars, base_structure=None):
    i = as_string.index("=")
    name = as_string[:i]
    value = as_string[i + 1 :]
    if value[0] == "[":
        return name, get_feature_struct(value, semantic_structures, struct_vars, base_structure)
    elif "[" in value:
        # named struct
        struct_name = value[: value.index("[")]
        structure = get_feature_struct(value, semantic_structures, struct_vars, base_structure)
        if base_structure:
            for feature in base_structure["features"]:
                if "=" in feature:
                    separator = feature.index("=")
                    if feature[:separator] in structure["features"]:
                        continue
                    structure["features"][feature[:separator]] = {"type": "var", "name": feature[separator + 1 :]}
                else:
                    if feature in structure["features"]:
                        continue
                    structure["features"][feature] = {"type": "var", "name": struct_name + feature.capitalize()}
        struct_vars[struct_name] = structure
        return name, structure
    elif value in struct_vars:
        return name, struct_vars[value]
    else:
        return name, {"type": "var", "name": value}


def has_base_structure(xmlelems, source_id, dictionaries, semantic_structures):
    for child in xmlelems:
        attributes = child.attrib
        if not "edge" in attributes:
            continue
        target = attributes["target"]
        if not attributes["source"] == source_id or not target in dictionaries:
            continue
        target = dictionaries[target]
        if not target["type"] == "semantic_structure":
            continue
        return semantic_structures[target["name"]]
    return None


def get_feature_struct(as_string, semantic_structures, struct_vars, base_structure):
    str2 = as_string[as_string.index("[") + 1 : -1]
    features = {}
    record = ""
    recording = True
    nested = 0
    for i in str2:
        if recording and nested == 0 and i == ",":
            recording = False
            sub_base_structure = None
            if base_structure and record[: record.index("=")] in base_structure["features"]:
                sub_base_structure = base_structure["features"][record[: record.index("=")]]
                if sub_base_structure["type"] == "var":
                    sub_base_structure = None
                else:
                    sub_base_structure = semantic_structures[sub_base_structure["reference"]]
            name, feature = translate_feature(record, semantic_structures, struct_vars, sub_base_structure)
            features[name] = feature
            record = ""
        if i == "[":
            nested += 1
        if i == "]":
            nested -= 1
        if not recording and (i != "," and i != " "):
            recording = True
        if recording:
            record += i

    sub_base_structure = None
    if base_structure and record[: record.index("=")] in base_structure["features"]:
        sub_base_structure = base_structure["features"][record[: record.index("=")]]
        if sub_base_structure["type"] == "var":
            sub_base_structure = None
        else:
            sub_base_structure = semantic_structures[sub_base_structure["reference"]]
    name, feature = translate_feature(record, semantic_structures, struct_vars, base_structure)
    features[name] = feature
    return {"type": "structure", "features": features}


def feature_struct_to_str(structure):
    s = ""
    first = True
    for f in structure["features"]:
        if first:
            first = False
        else:
            s += ","
        feature = structure["features"][f]
        if feature["type"] == "var":
            true_var = ""
            if feature["name"][0].isupper():
                true_var = "?"
            s += f + "=" + true_var + feature["name"]
        else:
            s += f + "=" + feature_struct_to_str(feature)
    return "[" + s + "]"
