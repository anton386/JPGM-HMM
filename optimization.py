import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize

class Optimization(object):

    def __init__(self, method, framework="EM"):
        if method == "newton":
            self.optimizer = self.newton_raphson
            self.framework = framework
        elif method == "BFGS":
            self.optimizer = self.BFGS

        self.DEBUG = False
    
    def newton_raphson(self, obj_func, jcb_func, hess_func, x0, 
                       step_factor=10.0, TOL=1e-4, NMAX=60):
        n = 1
        step_size = 1.0
        
        while n <= NMAX:
            
            nll = obj_func(x0)
            jcb = jcb_func(x0)
            hess = hess_func(x0)
            hess_inv = inv(hess)
            
            #DEBUG
            if self.DEBUG:
                print "=================="
                print "  Newton Raphson"
                print "=================="
                print "Iteration #%s" % n
                print "Step Size: %s" % step_size
                print "Current Coordinates: %s" % x0
                print "Current Negative-Likelihood: %s" % nll
                
                print "Jacobian:"
                print jcb
                print "Hessian:"
                print hess
                print "Hessian Inverse:"
                print hess_inv

            x1 = x0 - step_size * (np.dot(hess_inv, jcb))
            new_nll = obj_func(x1)
            new_jcb = jcb_func(x1)
            nll_diff = new_nll - nll
            relative_diff = abs(nll_diff/nll)
            
            #DEBUG
            if self.DEBUG:
                print "New Coordinates: %s" % x1
                print "New Negative-Likelihood: %s" % new_nll
                print "Difference: %s" % nll_diff
                print "Relative Diff: %s" % relative_diff
                print "Diff (step): %s" % np.abs(x1-x0)
                print "Diff (jcb): %s" % np.abs(new_jcb-jcb)
                print "================="

            """
            if self.framework == "EM":
                if np.isnan(new_nll) or nll_diff > 0.0:
                    step_size = step_size/step_factor
                else:
                    is_step_converged = np.all(np.abs(x1-x0) <= TOL)
                    is_func_converged = np.all(np.abs(new_jcb-jcb) <= FUNC_TOL)
                    if is_step_converged or is_func_converged:
                        return (True, x1, new_nll)
                    else:  # keep going
                        x0 = x1
                        n += 1
            """
            
            if self.framework == "EM":
                if np.isnan(new_nll) or nll_diff > 0.0:  # change step size
                    step_size = step_size/step_factor
                else:
                    if relative_diff > TOL:  # keep going
                        x0 = x1
                    else:  # success
                        return (True, x1, new_nll)
                    n+=1            
            elif self.framework == "ENR":
                if np.isnan(new_nll) or nll_diff > 0.0:  # change step size
                    step_size = step_size/step_factor
                    n+=1
                else:
                    return (True, x1, new_nll)

        return (False, x0, obj_func(x0))
        
    
    def BFGS(self, obj_func, jcb_func, x0, TOL=1e-4):
        res = minimize(obj_func, x0, jac=jcb_func, method="BFGS", tol=TOL)
        return (res.success, res.x, res.fun)
    

