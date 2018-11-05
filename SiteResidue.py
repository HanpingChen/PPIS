# author：chenhanping
# date 2018/11/1 下午1:34
# copyright ustc sse

class SiteResidue:
    """
    位点实体类
    """
    def __init__(self):
        self.id = ('A', (' ', 1, ' '))
        self.chain_id = 'A'
        self.acc = 0
        self.chain_index = 1
        self.seq = 'S'
        self.seq_index = self.chain_index

    def __str__(self):
        return self.chain_id + " " + self.seq + " " + str(self.chain_index) + " " + str(self.seq_index)