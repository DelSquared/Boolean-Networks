import numpy as np
cimport cython
# cimport numpy as np
#import pandas as pd
from cython.parallel import prange
from libc.stdlib cimport malloc, free

class BooleanNetwork:
  def __init__(self, A):
    self.A = A
  def set_init(self, init):
    self.init = init
  def set_operation(self, operation):
    self.operation = operation
  def simulate(self, T, py = False):
    if py:
      return calculate_py(self.A, self.init, T)
    else:
      return calculate_cy(self.A, self.init, T)

def calculate_py(A, init, T):
  N = len(init)
  result = np.zeros((T, N), dtype=np.int)
  result[0] = ((A @ init)%2)
  for t in range(T-1):
    result[t+1] = ((A @ result[t])%2)
  return result


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def calculate_cy(A, init, int T):
  cdef:
    int N = int(len(init))
    int[:,:] _A = A
    int[:] _init = init
    int[:,:] result = np.zeros((T, N), dtype=np.int)
  _calculate(&_A[0,0], &_init[0], N, T, &result[0,0])
  return np.ascontiguousarray(result)


cdef void _calculate(int* A, int* init, int N, int T, int* result):
  cdef:
    int t = 0
  _matmulmod2(A, init, result, N)
  for t in range(T-1):
    _matmulmod2(A, &result[N*t], &result[N*(t+1)], N)

cdef void _matmulmod2(int* A, int* x, int* y, int n):
  #__debug_print(x, n, 0)
  cdef:
    int i = 0
    int j = 0
  for i in prange(n, nogil = True):
    y[i] = 0
    for j in range(n):
      y[i] += x[j] * A[i + n*j]
    y[i] = y[i] % 2
  #__debug_print(y, n, 0)

# for testing only
cdef void __debug_print(int* x, int n, int m):
  if m == 0:
    print([x[i] for i in range(n)])
  else:
    print([[x[i + n*j] for i in range(n)] for j in range(m)])
