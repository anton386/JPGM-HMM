import sys
import math
import numpy as np
import scipy.stats as spstats
import argparse
import pprint as pp

from jrun import JPGMRun
from data import Data

from independent import IndependentModel

try:
    from hmm_optimized import OptimizedHMM as HMM
    print "Imported Optimized HMM"
except ImportError:
    from hmm import HMM
    print "Imported non-Optimized HMM"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", default=-1, type=int,
                        help="Process the first i number of transcripts")
    parser.add_argument("-p", 
                        help="Select the optimization method [newton|BFGS]")
    parser.add_argument("-m",
                        help="Select the model [independent|hmm]")
    parser.add_argument("-l", default=1.0, type=float,
                        help="Use transcripts with load greater than l")
    parser.add_argument("-o", 
                        help="Output of Hidden States")
    parser.add_argument("-t", default=1, type=int,
                        help="Number of Threads")
    parser.add_argument("counts", nargs="*",
                        help="Expected Counts Data")
    parser.add_argument("--opt_tol", default=1e-6, type=float,
                       help="Optimizer's Tolerance")
    parser.add_argument("--opt_nmax", default=60, type=int,
                        help="Maximum number of iterations for the optimization step")
    parser.add_argument("--em_tol", default=1e-6, type=float,
                        help="EM's tolerance")
    parser.add_argument("--em_nmax", default=60, type=int,
                        help="Maximum number of iterations for the EM step")
    parser.add_argument("--em_tol_method", default="relative",
                        help="Whether to use relative or absolute change in log-likelihood")
    parser.add_argument("-d",
                        help="Data file")
    parser.add_argument("-i",
                        help="ID file")
    parser.add_argument("--abundance", action="store_true",
                        help="Transcript abundance in ID file")
    parser.add_argument("--theta", default=None,
                        help="Theta is provided")
    parser.add_argument("--limit", default=1000, type=int,
                        help="Limit the size of the read count")
    parser.add_argument("--framework", default="EM",
                        help="Choose the appropriate framework [EM|ENR]")
    parser.add_argument("--seed", default=None, type=int,
                        help="Set the seed")
    parser.add_argument("--initialize_with_indpt", action="store_true",
                        help="Initialize HMM with independent values")
    parser.add_argument("--step", default=10.0, type=float,
                        help="Set the step size during EM optimization")

    args = parser.parse_args()
    
    if len(args.counts) > 0:
        datum = Data(max_transcript=args.n, 
                     load_threshold=args.l)
        datum.load_counts(args.counts)
    else:
        datum = Data(max_transcripts=args.n)
        if args.abundance:
            datum.load_abundance(args.i)
        datum.load_data(args.d, limit=args.limit)

    # ==========================================================================
    # Initialize with Independent
    # ==========================================================================
    if args.initialize_with_indpt:
        
        # Initialize Independent Model
        model = IndependentModel(datum.N, datum.index, datum.library_size,
                                 transcript_abundance=datum.transcript_abundance,
                                 seed=args.seed)
        model.randomize_theta(2, 2)
        
        # Initialize Run
        jrun = JPGMRun()
        jrun.run(model, args)

        init_theta = model.theta
        init_pZ = model.pZ

    # ==========================================================================
    # Real Runs
    # ==========================================================================
    if args.m == "independent":
        model = IndependentModel(datum.N, datum.index, datum.library_size, 
                                 transcript_abundance=datum.transcript_abundance, 
                                 seed=args.seed, threads=args.t)
    elif args.m == "hmm":
        model = HMM(datum.N, datum.index, datum.library_size, 
                    transcript_abundance=datum.transcript_abundance,
                    seed=args.seed, threads=args.t)

    if args.initialize_with_indpt:
        model.pZ = init_pZ
        model.theta = init_theta
    else:
        if args.theta != None:
            """ DEBUG theta values
            theta = [0.5, 0.2, 0.001, 0.10, 0.5]
            theta = [0.03, 0.002, 0.001, 0.001, 0.002]
            theta = [0.64098537, 0.07507497, 1.3072797, -1.23522459, 2.13067915]
            """
            model.theta = map(float, args.theta.split(","))
        else:
            model.randomize_theta(2, 2)

    jrun = JPGMRun()
    jrun.run(model, args)
    
    model.output_states(args.o)
