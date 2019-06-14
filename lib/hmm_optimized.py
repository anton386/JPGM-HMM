import sys
import numpy as np

cimport cython
from cython.parallel import prange, parallel

from libc.math cimport exp
from libc.math cimport log1p

from hmm import HMM

"""http://software.ligo.org/docs/lalsuite/lalinference/logaddexp_8h_source.html
cdef double lse(double x, double y):
    cdef double tmp
    tmp = x - y
    if tmp > 0:
        return x + log1p(exp(-tmp))
    elif tmp <= 0:
        return y + lop1p(exp(tmp))
    else:
        return x + y
"""

class OptimizedHMM(HMM):
    def __init__(self, N, index, library_size, transcript_abundance=None, theta=[], seed=None, threads=1):
        HMM.__init__(self, N, index, library_size, transcript_abundance, theta, seed, threads)
    
    # DONE
    @cython.boundscheck(False)
    def calc_forward(self, long i, long start, long end):
        self.initialize_forward_variable(i, start, end)
        
        cdef long j, k, k_
        cdef long K = self.K
        cdef long threads = self.threads
        
        cdef double [:, :] lm = self.log_model_cache[i-1]
        cdef double [:, :] forward = self.forward
        cdef double [:, :] log_pT = self.log_pT
        
        cdef double [:, :] M = np.zeros((end-start, self.K))
        cdef double [:, :] V = np.zeros((end-start, self.K))
        cdef double [:, :] TMP = np.zeros((end-start, self.K))
        
        for j in range(1, end-start):
            
            for k in range(K):
                
                M[j,k] = lm[k,j]
                V[j,k] = forward[j-1,0] + log_pT[0,k]
                
                for k_ in range(1, K):

                    TMP[j,k] = V[j,k] - (forward[j-1,k_] + log_pT[k_,k])
                    if TMP[j,k] > 0:
                        V[j,k] = V[j,k] + log1p(exp(-TMP[j,k]))
                    elif TMP[j,k] <= 0:
                        V[j,k] = (forward[j-1,k_] + log_pT[k_,k]) + log1p(exp(TMP[j,k]))
                    else:
                        V[j,k] = V[j,k] + (forward[j-1,k_] + log_pT[k_,k])
                    
                    #DEBUG V[j,k] = logaddexp(V[j,k], forward[j-1,k_] + log_pT[k_,k])
                
                forward[j,k] = M[j,k] + V[j,k]
    
    # DONE
    @cython.boundscheck(False)
    def calc_backward(self, long i, long start, long end):
        self.initialize_backward_variable(i, start, end)

        cdef long j, k, k_
        cdef long K = self.K
        cdef long threads = self.threads
        
        cdef double [:, :] lm = self.log_model_cache[i-1]
        cdef double [:, :] backward = self.backward
        cdef double [:, :] log_pT = self.log_pT

        cdef double [:, :] V = np.zeros((end-start-1, self.K))
        cdef double [:, :] TMP = np.zeros((end-start-1, self.K))
        cdef long [:] J = np.arange(end-start-1)[::-1]
        
        for j in range(end-start-1-1, -1, -1):

            for k in range(K):

                V[j,k] = (backward[j+1,0]
                          + log_pT[k,0]
                          + lm[0,j+1])
                
                for k_ in range(1, K):
                    TMP[j,k] = V[j,k] - (backward[j+1,k_] + log_pT[k,k_] + lm[k_,j+1])
                    if TMP[j,k] > 0:
                        V[j,k] = V[j,k] + log1p(exp(-TMP[j,k]))
                    elif TMP[j,k] <= 0:
                        V[j,k] = (backward[j+1,k_] + log_pT[k,k_] + lm[k_,j+1]) + log1p(exp(TMP[j,k]))
                    else:
                        V[j,k] = V[j,k] + (backward[j+1,k_] + log_pT[k,k_] + lm[k_,j+1])

                    #DEBUG V[J[j],k] = lse(V[J[j],k], (backward[J[j+1],k_] + log_pT[k,k_] + lm[k_,J[j+1]]))
                
                backward[j,k] = V[j,k]
