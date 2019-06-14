import numpy as np
from optimization import Optimization

class JPGMRun(object):

    def __init__(self):
        self.theta = []
        self.iteration = 1
        self.nll = np.inf
        self.success = None
        self.diff = None
        self.relative_diff = None
    
    def run(self, model, args):
        
        # ENR or EM
        if args.framework == "ENR":
            print "Currently Performing Expectation-Newton-Raphson"
        elif args.framework == "EM":
            print "Currently Performing Expectation-Maximization"
        
        self.theta = model.theta
        self.print_init(model)
        while True:
            
            try:

                # ================
                # Expectation Step
                # ================

                # Update Lambda values
                if not args.abundance:
                    model.get_ld(self.theta)
                
                # Get new pZ
                if type(model).__name__ == "IndependentModel":
                    model.get_pZ(self.theta)
                elif type(model).__name__ == "HMM":
                    model.get_pZ_and_pT(self.theta)
                    model.get_pi()
                elif type(model).__name__ == "OptimizedHMM":
                    model.get_pZ_and_pT(self.theta)
                    model.get_pi()
                    
                # Update phi values
                model.get_phi()

                self.nll = model.log_expected(self.theta)

                # =================
                # Maximization Step
                # =================
                if args.p == "BFGS":
                    o = Optimization(args.p, args.framework)
                    self.success, self.theta, new_nll = o.optimizer(model.log_expected,
                                                                    model.log_jacobian,
                                                                    self.theta, 
                                                                    TOL=args.opt_tol)
                elif args.p == "newton":
                    o = Optimization(args.p, args.framework)
                    self.success, self.theta, new_nll = o.optimizer(model.log_expected,
                                                                    model.log_jacobian,
                                                                    model.log_hessian,
                                                                    self.theta, 
                                                                    step_factor=args.step,
                                                                    TOL=args.opt_tol,
                                                                    NMAX=args.opt_nmax)
                
                
                # Update theta
                model.theta = self.theta
                
                # Calculate Difference
                self.diff = new_nll - self.nll

                if self.iteration <= args.em_nmax:
                    self.print_iter(model)
                else:
                    break

                if args.framework == "ENR":
                    if not self.success:
                        break

                if args.em_tol_method == "absolute":
                    if (self.diff >= -args.em_tol) and (self.diff < 0.0):
                        break
                elif args.em_tol_method == "relative":
                    if self.diff != -np.inf:
                        self.relative_diff = abs(self.diff) / abs(self.nll)
                        if (self.relative_diff <= args.em_tol):
                            break

                self.nll = new_nll

                self.iteration += 1
            except KeyboardInterrupt:
                break
        
        self.nll = new_nll
        self.print_final(model)
        

    def print_init(self, model):
        print "Initializing Parameters"
        print "================"
        print "  Initialization"
        print "================"
        print "Iteration: %s" % self.iteration
        print "    Theta: %s" % model.theta
        print "    Phi: %s" % model.phi
        if "HMM" in type(model).__name__ :
            print "    Pi: %s" % model.pi
            print "    Transition:\n%s" % model.pT
        print "================"

    def print_iter(self, model):
        print "================"
        print "  Maximization"
        print "================"
        print "Iteration: %s" % self.iteration
        print "    Status: %s" % self.success
        print "    Negative Log-Likelihood: %s" % self.nll
        print "    Theta: %s" % model.theta
        print "    Phi: %s" % model.phi
        print "    Diff: %s" % self.diff
        if "HMM" in type(model).__name__:
            print "    Pi: %s" % model.pi
            print "    Transition:\n%s" % model.pT
        print "================"

    def print_final(self, model):
        print "================="
        print "  Final"
        print "================="
        print "    Iteration: %s" % self.iteration
        print "    Negative Log-Likelihood: %s" % self.nll
        print "    Theta: %s" % model.theta
        print "    Phi: %s" % model.phi
        print "================="
