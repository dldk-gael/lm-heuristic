def revert(bindings):
    for i in range(len(bindings)):
        if isinstance(bindings[i], list):
            for feature in bindings[i][1]:
                del bindings[i][0].features[feature]
        else:
            binding = bindings[-i]
            binding.ref = None


def copy_features(body):
    variables = {}
    mirrors = []
    mirrors.append({"features": body["head_feature"].get_mirror(variables)})
    for i in range(len(body["body_symbols"])):
        mirrors.append({"str": body["body_symbols"][i], "features": body["body_features"][i].get_mirror(variables)})
    return mirrors

def is_symbol_terminal(symbol):
    return (symbol["str"].startswith('"') or symbol["str"].startswith("'"))

class PVar:
    def __init__(self, name):
        self.ref = None
        self.name = name
        self.isVar = True
        self.isConst = False
        self.isStruct = False

    def unify(self, other, bindings, expand_features=False):
        if self == other:
            return True
        if self.ref == None:
            self.ref = other
            bindings.append(self)
            return True
        else:
            return self.ref.unify(other, bindings, expand_features)

    def show(self):
        if self.ref == None:
            return self.name
        return self.ref.show()

    def __str__(self):
        refstr = "UNBOUND"
        if self.ref != None:
            refstr = self.ref.show()
        return "(VAR " + self.name + "=" + refstr + ")"

    def get_mirror(self, variables):
        if self.name not in variables:
            variables[self.name] = PVar(self.name)
        return variables[self.name]


class PConstant:
    def __init__(self, value):
        self.value = value
        self.isVar = False
        self.isConst = True
        self.isStruct = False

    def unify(self, other, bindings, expand_features=False):
        if other.isVar and other.ref == None:
            other.ref = self
            bindings.append(other)
            return True
        elif other.isVar:
            return self.unify(other.ref, bindings, expand_features)
        return other.isConst and other.value == self.value

    def show(self):
        return str(self.value)

    def __str__(self):
        return self.show()

    def get_mirror(self, variables):
        return self


class PStruct:
    def __init__(self, features):
        self.features = features
        self.isVar = False
        self.isConst = False
        self.isStruct = True

    def unify(self, other, bindings, expand_features=False):
        if other.isVar and other.ref == None:
            other.ref = self
            bindings.append(other)
            return True
        elif other.isVar:
            return self.unify(other.ref, bindings, expand_features)
        elif other.isStruct and not other.features:
            return True
        elif other.isStruct:
            to_check = [f for f in self.features if f in other.features]
            for f in to_check:
                if not self.features[f].unify(other.features[f], bindings, expand_features):
                    return False
            if expand_features:
                new_features = [f for f in other.features if f not in self.features]
                for f in new_features:
                    self.features[f] = other.features[f]
                bindings.append([self, new_features])
            return True
        return False

    def show(self):
        s = "["
        first = True
        for f in self.features:
            if first:
                first = False
            else:
                s += ", "
            s += f + "=" + self.features[f].show()
        return s + "]"

    def __repr__(self):
        return self.show()

    def get_mirror(self, variables):
        mirror_features = {}
        for f in self.features:
            mirror_features[f] = self.features[f].get_mirror(variables)
        return PStruct(mirror_features)
