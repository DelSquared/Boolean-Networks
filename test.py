import numpy as np
import pandas as pd
import booleannetworks as bn
from matplotlib import pyplot as plt

N = 20
T = 500

def random_adj(n, bias):
    A = np.zeros((n,n), dtype=np.int)
    portion = 100*bias
    for i in range(n):
        for j in range(i,n):
            c = np.random.choice([0 if i > portion else 1 for i in range(100)])
            A[i,j] = c
            A[j,i] = c
        A[i,i] = 0
    return A

def K_adj(n):
    return np.ones((n,n), dtype=np.int) - np.eye(n, dtype=np.int)

A = random_adj(N, 0.5)

B = bn.BooleanNetwork(A)

init = np.array([np.random.choice([0,1]) for i in range(N)], dtype=np.int)

B.set_init(init)

print(A)
print()
print('', init)

plt.imshow(B.simulate(T))
plt.show()





