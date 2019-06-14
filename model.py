import sys
import math
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.special import gammaln
from scipy.special import polygamma
from scipy.misc import factorial
import scipy.stats as spstats


class Model(object):

    def __init__(self, N, index, library_size, transcript_abundance=None, theta=[], seed=None, threads=1):
        self.N = N
        self.index = index
        self.delta = None
        self.pZ = []
        self.phi = []
        self.ld = transcript_abundance
        self.theta = theta
        self.seed = seed
        self.threads = threads

        self.T = 2
        self.K = 3  # including unknown

        # Initialize all Model-specific parameters
        self.initialize_delta(library_size)
        self.initialize_pZ()
        self.get_phi()

    def get_kappa_beta(self, theta):
        kappa, beta_12, beta_21, beta_22 = theta
        beta = [[0.0, beta_12],
                [beta_21, beta_22]]
        return kappa, np.array(beta)

    def output_states(self, out):
        with open(out, "w") as f_out:

            # header with parameter values
            f_out.write("# k*: %.5f\n" % self.theta[0])
            f_out.write("# beta: %.5f|%.5f|%.5f\n" % tuple(self.theta[1:]))
            

            for k, v in self.index.items():
                start, end = v
                for j, (x, y, z, l) in enumerate(zip(self.N[0][start:end], self.N[1][start:end], 
                                                     self.pZ[start:end], self.ld[start:end]), 1):
                    f_out.write("%s\n" % "\t".join([str(k), str(j), 
                                                    str(x), str(y), 
                                                    str(np.argmax(z)),
                                                    "\t".join(["%.3f"] * len(z)) % tuple(z),
                                                    "%.2f" % l]))

    # DONE - deal with (T,K)
    def randomize_theta(self, t, k):
        if self.seed:
            np.random.seed(self.seed)
        self.theta = np.random.random(1+((t*k)-1))
    
    # DONE
    def initialize_delta(self, library_size):
        delta = [0.0, 0.0]        
        delta[1] = np.log(float(library_size[1])/float(library_size[0]))
        self.delta = np.array(delta)

    # DONE
    def initialize_ld(self, transcript_abundance):
        self.ld = np.array(transcript_abundance)

    # DONE - deal with (T,K)
    def initialize_pZ(self, alpha=10):
        # dirichlet distribution
        d = spstats.dirichlet(tuple([alpha] * self.K), self.seed)
        self.pZ = d.rvs(self.N.shape[1])
    
    # TOREMOVE
    def calculate_TPM(self):
        ld_V1 = []
        ld_S1 = []
        for i in xrange(len(self.N[0])):
            ld_V1.append(self.N[0][i].sum() / float(len(self.N[0][i])))
            ld_S1.append(self.N[1][i].sum() / float(len(self.N[1][i])))
        
        ld_V1 = np.array(ld_V1)
        ld_S1 = np.array(ld_S1)
        
        TPM_V1 = (10.0**6 * ld_V1)/ld_V1.sum()
        TPM_S1 = (10.0**6 * ld_S1)/ld_S1.sum()
        
        self.TPM = (TPM_V1 + TPM_S1)/2.0
    
    # DONE - ld for every element
    def get_ld(self, theta):

        kappa, beta = self.get_kappa_beta(theta)
        
        new_ld = []

        N = self.N.sum(axis=0)
        Z = self.pZ[:,0] + self.pZ[:,1]

        num_1 = (self.pZ[:,0] * N) - (self.pZ[:,1] * kappa)
        den_1 = 1.0 + np.exp(self.delta[1] + beta[1][0])
        num_2 = (self.pZ[:,1] * N) - (self.pZ[:,0] * kappa)
        den_2 = np.exp(beta[0][1]) + np.exp(self.delta[1] + beta[1][1])
        num_3 = Z + N
        den_3 = den_1 * den_2
        
        for k, (start, end) in sorted(self.index.items()):
            a_i1 = (num_1[start:end].mean()/den_1) + (num_2[start:end].mean()/den_2)
            a_i2 = kappa * (num_3[start:end].mean()/den_3)
            ld = a_i1/2.0 + np.sqrt(np.square(a_i1)/4.0 + a_i2)
            
            new_ld += [ ld for _ in range(end-start) ]

        self.ld = np.array(new_ld)
    

    # DONE - deal with (T,K)
    def get_phi(self):

        phi = np.argmax(self.pZ, axis=1)
        self.phi = np.array([np.count_nonzero(phi == k) for k in xrange(self.K) ]) / float(len(phi))
    
    # DONE
    def model(self, kappa, beta):
        lm = self.log_model(kappa, beta)
        c = np.array([ np.log(np.arange(1, m+1)).sum()
                       + np.log(np.arange(1, n+1)).sum() for m, n in zip(self.N[0], self.N[1]) ])
        return np.exp(lm - c)

    # DONE - deal with (T, K)
    def log_model(self, kappa, beta):
        
        N = self.N.sum(axis=0)
        lm = []
        
        lld = np.log(self.ld)
        for k in range(self.K-1):
            
            c_1K = self.ld * np.exp(self.delta[0] + beta[0][k])
            c_2K = self.ld * np.exp(self.delta[1] + beta[1][k])
            
            l = (gammaln(N + kappa) - gammaln(kappa))
            l += kappa * np.log(kappa) 
            l += self.N[0] * (lld + self.delta[0] + beta[0][k])
            l += self.N[1] * (lld + self.delta[1] + beta[1][k])
            l -= (N + kappa) * (np.log(c_1K + c_2K + kappa))
            
            lm.append(l)
        return np.array(lm)
    

    # DONE - deal with (T, K)
    def first_drvt_log_model_kappa(self, kappa, beta):
        
        N = self.N.sum(axis=0)
        J = []
        for k in range(self.K-1):
            
            pgamma = polygamma(0, N + kappa) - polygamma(0, kappa)
            
            c_1K = self.ld * np.exp(self.delta[0] + beta[0][k])
            c_2K = self.ld * np.exp(self.delta[1] + beta[1][k])
            
            J_k = pgamma
            J_k += 1.0 + np.log(kappa)
            J_k -= np.log(c_1K + c_2K + kappa)
            J_k -= (N + kappa)/(c_1K + c_2K + kappa)
            
            J.append(J_k)
            
        return np.array(J)
    
    # DONE - deal with (T, K)
    def first_drvt_log_model_beta(self, kappa, beta):

        N = self.N.sum(axis=0)

        J = np.zeros((self.T * (self.K-1), self.N.shape[1]))
        for t in range(self.T):
            for k in range(self.K-1):

                if t != 0 or k != 0:
                    
                    c_1K = self.ld * np.exp(self.delta[0] + beta[0][k])
                    c_2K = self.ld * np.exp(self.delta[1] + beta[1][k])
                    c_TK = self.ld * np.exp(self.delta[t] + beta[t][k])

                    J_b = self.N[t] - ((N + kappa)*(c_TK)/(c_1K + c_2K + kappa))
                    
                    J[(t*self.T)+k] = J_b
        
        return J

    # DONE - deal with (T, K)
    def second_drvt_log_model_kappa_kappa(self, kappa, beta):

        N = self.N.sum(axis=0)
        
        H = []
        for k in range(self.K-1):
            
            pgamma = polygamma(1, N + kappa) - polygamma(1, kappa)

            c_1K = self.ld * np.exp(self.delta[0] + beta[0][k])
            c_2K = self.ld * np.exp(self.delta[1] + beta[1][k])

            num = c_1K + c_2K - N
            denom = c_1K + c_2K + kappa
            
            H_kk = pgamma 
            H_kk += 1.0/kappa
            H_kk -= 1.0/denom
            H_kk -= num/np.square(denom)

            H.append(H_kk)
        
        return np.array(H)

    # DONE - deal with (T, K)
    def second_drvt_log_model_beta_kappa(self, kappa, beta):

        TK = []
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    TK.append((t, k))

        N = self.N.sum(axis=0)

        H = []
        for u in range(len(TK)):
            t, k = TK[u]
            c_TK = self.ld * np.exp(self.delta[t] + beta[t][k])

            num_1 = c_TK
            num_2 = c_TK * (N + kappa)
            den_1 = kappa
            for t_ in xrange(self.T):
                den_1 += self.ld * np.exp(self.delta[t_] + beta[t_][k])
        
            H_bk = -(num_1/den_1) + (num_2/np.square(den_1))
            
            H.append(H_bk)
        
        return np.array(H)

    # DONE - deal with (T, K)
    def second_drvt_log_model_beta_beta(self, kappa, beta):

        TK = []
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    TK.append((t, k))

        N = self.N.sum(axis=0)
        H = np.zeros((len(TK), len(TK), self.N.shape[1]))
        
        for u in range(len(TK)):
            for v in range(u+1):

                t1, k1 = TK[u]
                t2, k2 = TK[v]
                
                if k1 == k2:
                    if t1 != t2:
                        
                        c_TK = np.square(self.ld) * np.exp(self.delta[t1] + beta[t1][k1] + self.delta[t2] + beta[t2][k2])
                        num_1 = (N + kappa) * c_TK
                        den_1 = kappa
                        for t_ in xrange(self.T):
                            den_1 += self.ld * np.exp(self.delta[t_] + beta[t_][k1])
                            
                        H[u, v] = -(num_1/np.square(den_1))
                        
                    elif t1 == t2:
                        
                        c_TK = self.ld * np.exp(self.delta[t1] + beta[t1][k1])
                        num_1 = (N + kappa) * c_TK
                        den_1 = kappa
                        for t_ in xrange(self.T):
                            den_1 += self.ld * np.exp(self.delta[t_] + beta[t_][k1])
                        num_2 = self.ld * np.exp(self.delta[t2] + beta[t2][k2]) - den_1
                        
                        H[u, v] = (num_1 * num_2)/(np.square(den_1))

        return H

    # DONE - deal with (T,K)
    def log_expected(self, theta):
        
        kappa, beta = self.get_kappa_beta(theta)

        log_total = 0.0
        
        lm = self.log_model(kappa, beta)

        for k in range(self.K-1):
            log_total += (self.pZ[:,k] * lm[k]).sum()
            
        return -log_total

    # DONE - deal with (T,K)
    def log_jacobian(self, theta):
        
        kappa, beta = self.get_kappa_beta(theta)
        log_total = [0.0, 0.0, 0.0, 0.0]
        
        J_k = self.first_drvt_log_model_kappa(kappa, beta)
        J_b = self.first_drvt_log_model_beta(kappa, beta)
        
        # Kappa (1 value)
        for k in range(self.K-1):
            log_total[0] += (self.pZ[:,k] * J_k[k]).sum()
                    
        # Beta (k x t values)
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    log_total[(t*2)+k] += (self.pZ[:,k] * J_b[(t*2)+k]).sum()
        
        log_total = np.array(log_total)
        
        # DEBUG
        # print "Jacobian %s" % -log_total
        return -log_total

    # DONE - deal with (T,K)
    def log_hessian(self, theta):

        kappa, beta = self.get_kappa_beta(theta)
        
        p = 1 + ((self.T * (self.K-1)) - 1)
        log_total = np.zeros((p,p)).tolist()
        
        # Kappa Kappa
        H_kk = self.second_drvt_log_model_kappa_kappa(kappa, beta)
        for k in range(self.K-1):
            log_total[0][0] += (self.pZ[:,k] * H_kk[k]).sum()

        # Beta
        TK = []
        for t in range(self.T):
            for k in range(self.K-1):
                if t != 0 or k != 0:
                    TK.append((t,k))

        # Beta Kappa
        H_bk = self.second_drvt_log_model_beta_kappa(kappa, beta)
        for u in range(len(TK)):
            t, k = TK[u]
            d2FdBTKdK = (self.pZ[:,k] * H_bk[((t*self.T)+k)-1]).sum()
            log_total[u+1][0] += d2FdBTKdK
            log_total[0][u+1] += d2FdBTKdK

        
        # Beta Beta
        H_bb = self.second_drvt_log_model_beta_beta(kappa, beta)
        for u in range(len(TK)):
            for v in range(u+1):
                t1, k1 = TK[u]
                t2, k2 = TK[v]
                d2FdBTKdBTK = (self.pZ[:,k1] * H_bb[u,v]).sum()
            
                if u == v:
                    log_total[u+1][v+1] += d2FdBTKdBTK
                else:
                    log_total[u+1][v+1] += d2FdBTKdBTK
                    log_total[v+1][u+1] += d2FdBTKdBTK

        log_total = np.array(log_total)
        return -log_total
