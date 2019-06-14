import sys
import math
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammaln
from scipy.special import polygamma
from scipy.misc import factorial
import scipy.stats as spstats

cimport cython
from cython.parallel import prange, parallel
from cython_gsl cimport *

from libc.math cimport log
from libc.math cimport exp

from model import Model

class OptimizedModel(Model):

    def __init__(self, N, index, library_size, transcript_abundance=None, theta=[], seed=None, threads=1):
        
        Model.__init__(self, N, index, library_size, transcript_abundance, theta, seed, threads)

        self.threads = threads
    
    # DONE
    @cython.boundscheck(False)
    def model(self, double kappa, double [:, :] beta):

        M = np.zeros((self.K-1, self.N.shape[1]))
        
        cdef long i, j, k
        cdef long K = self.K
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:, :] N = self.N
        cdef double [:, :] lm = self.log_model(kappa, beta)
        cdef double [:] c = np.zeros(n)
        cdef double [:, :] m = M
        
        #for i in range(n):
        for i in prange(n, schedule="static", nogil=True, num_threads=threads):
            
            for j in range(N[0,i]):
                c[i] += log(j+1)
                
            for j in range(N[1,i]):
                c[i] += log(j+1)

            for k in range(K-1):
                m[k,i] = exp(lm[k,i] - c[i])

        return M

    # DONE
    @cython.boundscheck(False)
    def log_model(self, double kappa, double [:, :] beta):

        sum_of_N = self.N.sum(axis=0)
        LM = np.zeros((self.K-1, self.N.shape[1]))

        cdef long i, k
        cdef long K = self.K
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:, :] N = self.N
        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :] lm = LM
        
        for k in range(K-1):
            
            #for i in range(n):
            for i in prange(n, schedule="static", nogil=True, num_threads=threads):
                
                # DEBUG c_1K = ld[i] * exp(delta[0] + beta[0,k])
                # DEBUG c_2K = ld[i] * exp(delta[1] + beta[1,k])

                lm[k,i] = gsl_sf_lngamma(s_N[i] + kappa) - gsl_sf_lngamma(kappa)
                lm[k,i] += kappa * log(kappa)
                lm[k,i] += N[0,i] * (log(ld[i]) + delta[0] + beta[0,k])
                lm[k,i] += N[1,i] * (log(ld[i]) + delta[1] + beta[1,k])
                lm[k,i] -= (s_N[i] + kappa) * (log((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) + kappa))

        return LM
    
    # DONE
    @cython.boundscheck(False)
    def first_drvt_log_model_kappa(self, double kappa, double [:, :] beta):

        sum_of_N  = self.N.sum(axis=0)
        J_k = np.zeros((self.K-1, self.N.shape[1]))

        cdef long i, k
        cdef long K = self.K
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:, :] N = self.N
        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :] J = J_k
        
        for k in range(K-1):

            #for i in range(n):
            for i in prange(n, schedule="static", nogil=True, num_threads=threads):

                #DEBUG c_1K = ld[i] * exp(delta[0] + beta[0,k])
                #DEBUG c_2K = ld[i] * exp(delta[1] + beta[1,k])

                J[k,i] = gsl_sf_psi_n(0, s_N[i] + kappa) - gsl_sf_psi_n(0, kappa)
                J[k,i] += 1.0 + log(kappa)
                J[k,i] -= log((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) + kappa)
                J[k,i] -= (s_N[i] + kappa)/((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) + kappa)

        return J_k
    
    # DONE
    @cython.boundscheck(False)
    def first_drvt_log_model_beta(self, double kappa, double [:, :] beta):

        sum_of_N = self.N.sum(axis=0)
        J_b = np.zeros((self.T * (self.K-1), self.N.shape[1]))

        cdef long i, k, t
        cdef long T = self.T
        cdef long K = self.K
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:, :] N = self.N
        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :] J = J_b
        
        for t in range(T):
            for k in range(K-1):

                if t != 0 or k != 0:
                    
                    #for i in range(n):
                    for i in prange(n, schedule="static", nogil=True, num_threads=threads):
        
                        # DEBUG c_1K = ld[i] * exp(delta[0] + beta[0,k])
                        # DEBUG c_2K = ld[i] * exp(delta[1] + beta[1,k])
                        # DEBUG c_TK = ld[i] * exp(delta[t] + beta[t,k])
                        
                        J[(t*T)+k, i] = N[t,i] - ((s_N[i] + kappa) * 
                                                  (ld[i] * exp(delta[t] + beta[t,k]))) / ((ld[i] * exp(delta[0] + beta[0,k])) 
                                                                                          + (ld[i] * exp(delta[1] + beta[1,k])) + kappa)
        
        return J_b

    # DONE
    @cython.boundscheck(False)
    def second_drvt_log_model_kappa_kappa(self, double kappa, double [:, :] beta):

        sum_of_N = self.N.sum(axis=0)
        H_kk = np.zeros((self.K-1, self.N.shape[1]))

        cdef long i, k
        cdef long T = self.T
        cdef long K = self.K
        cdef long t = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:, :] N = self.N
        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :] H = H_kk
        
        for k in range(K-1):
            
            #for i in range(n):
            for i in prange(n, schedule="static", nogil=True, num_threads=t):
                
                # DEBUG c_1K = ld[i] * exp(delta[0] + beta[0,k])
                # DEBUG c_2K = ld[i] * exp(delta[1] + beta[1,k])
                
                # DEBUG num = c_1K + c_2K - s_N[i]
                # DEBUG denom = c_1K + c_2K + kappa
                
                H[k,i] = (gsl_sf_psi_n(1, s_N[i] + kappa) - gsl_sf_psi_n(1, kappa)
                          + 1.0/kappa
                          - 1.0/((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) - s_N[i])
                          - (((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) - s_N[i]) / 
                             (((ld[i] * exp(delta[0] + beta[0,k])) + (ld[i] * exp(delta[1] + beta[1,k])) + kappa) ** 2)))
        
        return H_kk

    # DONE
    @cython.boundscheck(False)
    def second_drvt_log_model_beta_kappa(self, double kappa, double [:, :] beta):
        
        sum_of_N = self.N.sum(axis=0)
        H_bk = np.zeros((self.T * (self.K-1), self.N.shape[1]))

        cdef long i, t, k
        cdef long T = self.T
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]
        
        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :] H = H_bk
        
        
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    
                    #for i in range(n):
                    for i in prange(n, schedule="static", nogil=True, num_threads=threads):

                        H[((t*T)+k)-1, i] = (-(ld[i] * exp(delta[t] + beta[t,k])) / 
                                              (kappa 
                                               + (ld[i] * exp(delta[0] + beta[0,k])) 
                                               + (ld[i] * exp(delta[1] + beta[1,k])))
                                              + ((ld[i] * exp(delta[t] + beta[t,k]) * (s_N[i] + kappa)) / 
                                                 (kappa 
                                                  + (ld[i] * exp(delta[0] + beta[0,k]))
                                                  + (ld[i] * exp(delta[1] + beta[1,k])))**2))

        return H_bk

    # DONE
    @cython.boundscheck(False)
    def second_drvt_log_model_beta_beta(self, double kappa, double [:, :] beta):

        TK = []
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    TK.append((t, k))

        sum_of_N = self.N.sum(axis=0)
        H_bb = np.zeros((len(TK), len(TK), self.N.shape[1]))
        
        cdef long i, u, v
        cdef long t1, k1, t2, k2
        cdef long threads = self.threads
        cdef Py_ssize_t n = self.N.shape[1]

        cdef long [:] s_N = sum_of_N
        cdef double [:] ld = self.ld
        cdef double [:] delta = self.delta
        cdef double [:, :, :] H = H_bb
        
        for u in range(len(TK)):
            for v in range(u+1):

                t1, k1 = TK[u]
                t2, k2 = TK[v]
                
                if k1 == k2:
                    if t1 != t2:

                        #for i in range(n):
                        for i in prange(n, schedule="static", nogil=True, num_threads=threads):
                            
                            H[u,v,i] = -( ( (s_N[i] + kappa) * (ld[i] ** 2) * exp(delta[t1] + beta[t1,k1] + delta[t2] + beta[t2,k2]) ) /
                                          ( (kappa + (ld[i] * exp(delta[0] + beta[0,k1])) + (ld[i] * exp(delta[1] + beta[1,k1]))) ** 2 ) )
                            
                    else:
                        
                        #DEBUG num_1 = ( (s_N[i] + kappa) * ld[i] * exp(delta[t1] + beta[t1,k1] + delta[t2] + beta[t2,k2]) )
                        #DEBUG num_2 = ( ld[i] * exp(delta[t2] + beta[t2,k2]) ) - ( (kappa + (ld[i] * exp(delta[0] + beta[0,k1])) + (ld[i] * exp(delta[1] + beta[1,k1]))) )
                        #DEBUG den_1 = ( (kappa + (ld[i] * exp(delta[0] + beta[0,k1])) + (ld[i] * exp(delta[1] + beta[1,k1]))) )

                        #for i in range(n):
                        for i in prange(n, schedule="static", nogil=True, num_threads=threads):
                            
                            H[u,v,i] = ( ( ( (s_N[i] + kappa) * ld[i] * exp(delta[t1] + beta[t1,k1] + delta[t2] + beta[t2,k2]) ) *
                                           ( ( ld[i] * exp(delta[t2] + beta[t2,k2]) ) - ( (kappa + (ld[i] * exp(delta[0] + beta[0,k1])) + (ld[i] * exp(delta[1] + beta[1,k1]))) ) ) ) /
                                         ( (kappa + (ld[i] * exp(delta[0] + beta[0,k1])) + (ld[i] * exp(delta[1] + beta[1,k1]))) ** 2) )
                            
        return H_bb
