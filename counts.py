class Counts(object):

    def __init__(self, f_name):
        self.f_name = f_name
        self.tid = None
        self.gid = None
        self.isoform_pct = None
        self.TPM = None
        self.FPKM = None
        self.EC = None
        self.AC = None
        self.counts = None
        self.row = None

        self.handle = self.__yield_lines()

    def __yield_lines(self):
        with open(self.f_name, "r") as f:
            for line in f:
                yield line.strip("\r\n")

    def next_row(self):
        self.row = self.handle.next().split("\t")
        
        self.tid = self.row[0]
        self.gid = self.row[1]
        self.isoform_pct = float(self.row[2])
        self.TPM = float(self.row[3])
        self.FPKM = float(self.row[4])
        self.EC = float(self.row[5])
        self.AC = float(self.row[6])
        self.counts = self.get_counts()
        
    def get_counts(self):
        return map(int, self.row[7].split(";"))

    def load(self):
        return sum(self.counts)/float(len(self.counts))
