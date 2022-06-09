import random
from matplotlib import pyplot as plt
from numpyFun import get_ones


def add(a, b):
    return a + b


# print(1,2)
def print_all(X):
    temp = [a * 10 for a in X]
    for i in temp:
        print(i)


# print_all([1,2,3,4])


m = random.random()

for i in range(10):
    print(i * 5)
# print(m)


fig1 = plt.figure(figsize=(5, 5))
X = [x * 2 for x in range(10)]
y = [i * 4 for i in range(10)]
plt.plot(X, y)
plt.xlabel('X')
plt.ylabel(y)
plt.show()

print(get_ones())
