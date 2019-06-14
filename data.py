from counts import Counts
import numpy as np
import gzip

class Data(object):
    
    def __init__(self, max_transcripts=5, load_threshold=1.0, t=2):
        self.N = []
        self.index = {}
        self.library_size = []
        self.transcript_abundance = None
        self.max_transcripts = max_transcripts
        self.load_threshold = load_threshold
        self.t = t
        
    def load_abundance(self, f_id):
        
        if f_id.endswith(".gz"):
            f = gzip.open(f_id, "rb")
        else:
            f = open(f_id, "r")
        
        transcript_abundance = []
        for no, line in enumerate(f):
            row = line.strip("\r\n").split("\t")
            TPM = map(float, row[2:])

            if sum(TPM) == 0.0:
                transcript_abundance.append(0.01)
            else:
                transcript_abundance.append(sum(TPM)/len(TPM))

            if no+1 == self.max_transcripts:
                break
        
        f.close()
                
        self.transcript_abundance = np.array(transcript_abundance)
    
    def initialize_counts(self):
        return [[] for i in range(self.t)]

    def initialize_library(self):
        self.library_size = np.sum(self.N, axis=1)
        
    def load_data(self, f_data, limit=1000):
        
        if f_data.endswith(".gz"):
            f = gzip.open(f_data, "rb")
        else:
            f = open(f_data, "r")
                
        curr_transcript = 1
        prev_index, curr_index = (0, 0)
        for line in f:
            row = line.strip("\r\n").split("\t")
            if curr_transcript != int(row[0]):
                if curr_transcript == self.max_transcripts:
                    break
                else:
                    self.index[curr_transcript] = (prev_index, curr_index)
                    curr_transcript += 1
                    prev_index = curr_index
            
            c = map(int, map(float, row[1:]))
            if c[0] >= limit or c[1] >= limit:
                divisor = max(c)/float(limit)
                c = [ int(n/divisor) for n in c ]
            
            self.N.append(c)
            curr_index += 1
        
        # one last time
        self.index[curr_transcript] = (prev_index, curr_index)

        # transpose and initialize library
        self.N = np.array(self.N).T
        self.initialize_library()
        
        f.close()
            
    """
    def load_data(self, f_data, limit=1000):
        
        if f_data.endswith(".gz"):
            f = gzip.open(f_data, "rb")
        else:
            f = open(f_data, "r")

        # initialize
        self.N = self.initialize_counts()
        self.library_size = [0.0 for i in range(self.t)]
        
        count = []
        current_transcript = 1

        for line in f:
            row = line.strip("\r\n").split("\t")

            if current_transcript != int(row[0]):
                if current_transcript == self.max_transcripts:
                    break

                count = np.array(count).T

                #TODO should we deal with transcripts < 10 in length
                for i in range(self.t):
                    self.library_size[i] += count[i].sum()
                    self.N[i].append(count[i])

                # reset here
                count = []
                current_transcript = int(row[0])

            # store here
            c = map(int, map(float, row[1:]))
            if c[0] >= limit or c[1] >= limit:
                divisor = max(c)/float(limit)
                c = [ int(n/divisor) for n in c ]
            count.append(c)

        # one last time
        count = np.array(count).T
        for i in range(self.t):
            self.N[i].append(count[i])

        f.close()
    """

    def load_counts(self, f_counts):
        i, j = (0, 0)
        o_counts = []
        for f_count in f_counts:
            o_counts.append(Counts(f_count))
            self.N.append([])
            self.library_size.append(0.0)

        while True:
            try:

                tid, load, abundance = ([], [], [])
                for n, o_count in enumerate(o_counts):
                    o_count.next_row()
                    tid.append(o_count.tid)
                    load.append(o_count.load())
                    abundance.append(o_count.TPM)
                    
                    # Get Global Parameters
                    self.library_size[n] += o_count.AC

                # test whether elements are identical
                if (tid.count(tid[0]) == len(tid) and 
                    np.count_nonzero(np.array(load) >= self.load_threshold) == len(load)):  
                    
                    for n, o_count in enumerate(o_counts):
                        self.N[n].append(np.array(o_count.counts))
                        
                    self.transcript_abundance.append(sum(abundance)/2.0)

                    j += 1
                    if j == self.max_transcripts:
                        break

                i += 1
                if i % 10000 == 0:
                    print i, j
            
            except StopIteration:
                break
