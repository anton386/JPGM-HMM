import numpy as np

try:
    from model_optimized import OptimizedModel as Model
    print "Imported Optimized Independent Model"
except ImportError:
    from model import Model
    print "Imported non-Optimized Independent Model"


class IndependentModel(Model):
    def __init__(self, N, index, library_size, transcript_abundance=None, theta=[], seed=None, threads=1):
        Model.__init__(self, N, index, library_size, transcript_abundance, theta, seed, threads)

    def calc_pZ(self, kappa, beta):
        new_pZ = []
        index_N1 = np.where(self.N[0] == 0)[0]
        index_N2 = np.where(self.N[1] == 0)[0]
        
        M = self.model(kappa, beta)

        for k in range(self.K):
            
            if k < (self.K-1):
                m = M[k]
            else:
                m = np.zeros(self.N.shape[1])
                m[np.intersect1d(index_N1, index_N2)] = 1.0
            
            new_pZ.append(self.pZ[:,k] * m)
        
        new_pZ = np.array(new_pZ).T

        # FUNKY things happen here - check for zeros
        if np.count_nonzero(new_pZ.sum(axis=1)[:,None] == 0) > 0:
            new_pZ[np.where(new_pZ.sum(axis=1)[:,None] == 0)[0]] = np.array([1, 1, 1])
        
        return new_pZ/new_pZ.sum(axis=1)[:,None]

    def get_pZ(self, theta):
        
        kappa, beta = self.get_kappa_beta(theta)

        pZ = self.calc_pZ(kappa, beta)
        self.pZ = pZ
        
