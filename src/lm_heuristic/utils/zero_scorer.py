class ZeroScorer:
    def build(self):
        return

    def __call__(self, nodes):
        return len(nodes) * [0.0]