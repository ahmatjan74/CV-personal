import numpy as np


def get_ones():
    return np.ones([3, 3])


class NumpyTestClass:
    _a = 1
    _b = 2

    def __int__(self):
        self._a = 2
        self._b = 4


if __name__ == '__main__':
    print(get_ones())
