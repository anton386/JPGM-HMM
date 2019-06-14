import sys
import numpy as np
import scipy.stats as spstats
import pprint as pp

try:
    from model_optimized import OptimizedModel as Model
    print "Imported Optimized HMM Model"
except ImportError:
    from model import Model
    print "Imported non-Optimized HMM Model"


class HMM(Model):
    def __init__(self, N, index, library_size, transcript_abundance=None, theta=[], seed=None, threads=1):
        Model.__init__(self, N, index, library_size, transcript_abundance, theta, seed, threads)
        self.pT = None
        self.log_model_cache = []
        self.model_cache = []
        self.forward = []
        self.backward = []
        self.fb = []
        self.xi = []
        self.pi = []
        
        # Initialize all HMM-specific parameters
        self.initialize_pT()
        self.initialize_xi()
        self.initialize_pi()

    # DONE
    def initialize_pT(self, alpha=10):

        # sample from dirichlet distribution
        d = spstats.dirichlet(tuple([alpha] * self.K), self.seed)
        self.pT = d.rvs(self.K)

    # DONE
    def initialize_xi(self):
        for i, (start, end) in sorted(self.index.items()):
            self.xi.append(np.ones((end-start, self.K, self.K)))
    
    # DONE
    def initialize_pi(self, alpha=10):
        d = spstats.dirichlet(tuple([alpha] * self.K), self.seed)
        self.pi = d.rvs()[0]
        #self.pi = np.ones(self.K)/float(self.K)

    # TOREMOVE
    def initialize_accurate_log_model_cache(self, i, kappa, beta):
        self.model_cache = []
        for k in range(3):
            if k < 2:
                pm = self.model(i, k, kappa, beta)
                pm[np.where(pm == 0.0)[0]] = 1e-50
                self.model_cache.append(pm)
            else:
                index_N1 = np.where(self.N[0][i] == 0)[0]
                index_N2 = np.where(self.N[1][i] == 0)[0]
                n = np.ones(len(self.N[0][i])) * 1e-50

                # choose either option
                n[np.intersect1d(index_N1, index_N2)] = 1.0
                self.model_cache.append(n)

        self.log_model_cache = np.log(self.model_cache)
    
    # DONE - one transcript at a time
    def initialize_log_model_cache(self, kappa, beta):
        log_model_cache = []
        
        LM = self.log_model(kappa, beta)

        for i, (start, end) in sorted(self.index.items()):
            
            lm = []
            for k in range(self.K):
                if k < (self.K-1):
                    lm.append(LM[k][start:end])
                else:
                    index_N1 = np.where(self.N[0][start:end] == 0)[0]
                    index_N2 = np.where(self.N[1][start:end] == 0)[0]
                    n = np.ones(end-start) * -50.0

                    # choose either option
                    n[np.intersect1d(index_N1, index_N2)] = 0.0
                    
                    lm.append(n)
            
            log_model_cache.append(np.array(lm))

        self.log_model_cache = log_model_cache

    # TODO deal with i (has to be controlled and not user-given)
    def initialize_forward_variable(self, i, start, end):
        self.forward = np.ones((end-start, self.K))
        
        for k_ in range(self.K):
            self.forward[0][k_] = np.log(self.pi[k_]) + self.log_model_cache[i-1][k_][0]

    # DONE
    def initialize_backward_variable(self, i, start, end):
        self.backward = np.ones((end-start, self.K))
        for k_ in range(self.K):
            self.backward[-1][k_] = np.log(1.0)
            #self.backward[-1][k_] = self.log_pT[k_][0]
            #self.backward[-1][k_] = self.log_pT[0][k_]
    
    # TODO deal with i (has to be controlled and not user-given)
    def forward_variable(self, i, j, k):
        
        """Alternative (Not much faster)
        m = np.tile(self.forward[j-1], (3,1)) + self.log_pT.T
        return self.log_model_cache[:,j] + np.logaddexp(np.logaddexp(m[:,0], m[:,1]), m[:,2])
        """
        
        M = self.log_model_cache[i-1][k][j]
        V = self.forward[j-1][0] + self.log_pT[0][k]
        for k_ in xrange(1, self.K):
            V = np.logaddexp(V, self.forward[j-1][k_] + self.log_pT[k_][k])
        return M + V
        
    # TODO deal with i (has to be controlled and not user-given)
    def backward_variable(self, i, j, k):
        V = (self.backward[j+1][0]
             + self.log_pT[k][0]
             + self.log_model_cache[i-1][0][j+1])
        for k_ in xrange(1, self.K):
            V = np.logaddexp(V, (self.backward[j+1][k_]
                                 + self.log_pT[k][k_]
                                 + self.log_model_cache[i-1][k_][j+1]))
        return V

    # DONE
    def calc_forward(self, i, start, end):
        self.initialize_forward_variable(i, start, end)
        for j in xrange(1, end-start):
            for k in range(self.K):
                f_ijk = self.forward_variable(i, j, k)
                self.forward[j][k] = f_ijk
            """Alternative
            self.forward[j] = self.forward_variable(j)
            """
        
    # DONE
    def calc_backward(self, i, start, end):
        self.initialize_backward_variable(i, start, end)
        for j in range(end-start-1)[::-1]:
            for k in range(self.K):
                b_ijk = self.backward_variable(i, j, k)
                self.backward[j][k] = b_ijk

    # DONE
    def calc_pZ_i(self, kappa, beta):
        
        rescale = self.fb.max(axis=1)[:,None]
        
        """
        # FUNKY things happen here - check for zeros
        if np.count_nonzero(new_pZ_i.sum(axis=1)[:,None] == 0) > 0:
            new_pZ_i[np.where(new_pZ_i.sum(axis=1)[:,None] == 0)[0]] = np.array([1, 1, 1])
        """
        
        # rescale
        new_pZ_i = np.exp(self.fb - rescale)/np.exp(self.fb - rescale).sum(axis=1)[:,None]
        return new_pZ_i

    # TODO deal with i (has to be controlled and not user-given)
    def calc_xi_i(self, i):

        """CORRECT
        """
        # xi
        s = np.logaddexp(np.logaddexp(self.fb[1][0], self.fb[1][1]), self.fb[1][2])
        v = (self.forward[:-1][:,:,None]
             + self.backward[1:].reshape((-1,1,3))
             + self.log_pT
             + self.log_model_cache[i-1].T[1:].reshape((-1,1,3)))
        
        new_xi_i = np.exp(v - s)

        """Experimental
        print self.forward[j:j+2][:,:,None]
        print self.backward[j+1:j+1+2].reshape((2,1,3))
        print self.log_model_cache.T[j+1:j+1+2].reshape((2,1,3))
        v = (self.forward[j:j+2][:,:,None] 
             + self.backward[j+1:j+1+2].reshape((2,1,3)) 
             + self.log_pT 
             + self.log_model_cache.T[j+1:j+1+2].reshape((2,1,3)))

        r = self.fb[j:j+2]
        p = np.logaddexp(np.logaddexp(r[:,0], r[:,1]), r[:,2])
        print np.exp(v - p[0])
        """
        
        """SLOW
        new_xi_i = np.zeros((len(self.N[0][i])-1, 3, 3))
        for j in range(len(self.N[0][i])-1):

            
            for k1 in range(3):
                for k2 in range(3):
                    new_xi_i[j][k1][k2] = (self.forward[j][k1] 
                                           + self.backward[j+1][k2]
                                           + self.log_pT[k1][k2]
                                           + self.log_model_cache[k2][j+1])
            
            '''faster
            new_xi_i[j] = self.forward[j][:,None] + self.backward[j+1] + self.log_pT + self.log_model_cache[:,j+1]
            '''
            
            s = np.logaddexp(np.logaddexp(self.fb[j][0], self.fb[j][1]), self.fb[j][2])
            new_xi_i[j] = np.exp(new_xi_i[j] - s)
        """

        """DEBUG - Unit Test
        for i in range(1000):
            print new_xi_i[i].sum()
        assert False
        """

        return new_xi_i

    # DONE
    def get_pZ_and_pT(self, theta):

        kappa, beta = self.get_kappa_beta(theta)
        
        self.log_pT = np.log(self.pT)
        
        # calculate stored values
        self.initialize_log_model_cache(kappa, beta)
        
        for i, (start, end) in sorted(self.index.items()):
            
            # forward algorithm
            self.calc_forward(i, start, end)

            # backward algorithm
            self.calc_backward(i, start, end)

            self.fb = self.forward + self.backward

            self.pZ[start:end] = self.calc_pZ_i(kappa, beta)
            
            self.xi[i-1] = self.calc_xi_i(i)
            
        self.get_pT()

    # DONE
    def get_pi(self):
        new_pi = np.array([0.0, 0.0, 0.0])

        for i, (start, end) in sorted(self.index.items()):
            new_pi += self.pZ[start:end][0]
        
        self.pi = new_pi / float(len(self.index))

    # DONE - deal with i (has to be controlled and not user-given)
    def get_pT(self):
        
        new_pT = np.array(np.zeros((self.K, self.K)))

        for i, (start, end) in sorted(self.index.items()):
            new_pT_i = np.array(np.zeros((self.K, self.K)))
            for j in xrange(end-start-1):
                new_pT_i += self.xi[i-1][j]

            new_pT += new_pT_i

        self.pT = new_pT/new_pT.sum(axis=1)[:,None]
