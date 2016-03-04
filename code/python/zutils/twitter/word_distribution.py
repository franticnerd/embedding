from math import log
import copy

class Distribution:

    def __init__(self, length=None):
        self.L = length
        self.data = {}

    # add the value for the given dimension
    def add_value(self, dim, value):
        old_value = self.data.get(dim, set())
        old_value.add(value)
        self.data[dim] = old_value

    def get_l1_norm(self):
        ret = 0
        for key in self.data:
            ret += len(self.data[key])
        return ret

    def normalize(self):
        l1_norm = self.get_l1_norm()
        for key in self.data:
            self.data[key] = float(len(self.data[key])) / float(l1_norm)

    # get the entroy for the probability distribution encoded by current vector
    def get_entropy(self):
        ret = 0
        self.normalize()
        for value in self.data.values():
            if value <= 1e-20:
                continue
            ret -= value * log(value)
        return ret

    # convert to dict object
    def to_dict(self):
        ret = copy.deepcopy(self.data)
        ret['L'] = self.L
        return ret

    def load_from_dict(self, d):
        self.L = d['L']
        self.data = copy.deepcopy(d)
        del self.data['L']


if __name__ == '__main__':
    pass
