import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Dianxian(Dataset):
    def __init__(self, fdata, ddata, t1, labels, excel):
        super(Dianxian, self).__init__()
        self.nodes = fdata
        self.edges = ddata
        self.t1 = t1
        self.labels = labels
        self.excel = excel

    def __getitem__(self, item):
        node = self.nodes[item]
        edge = self.edges[item]
        t1 = self.t1[item]
        label = self.labels[item]
        excel = self.excel[item]
        return node, edge, t1, label, excel

    def __len__(self):
        return self.nodes.shape[0]


def load_data(seed):
    # Load data
    fdata = process_fmri_data()
    ddata = process_dti_data()
    t1, labels, excel = process_other_data()
    
    # Shuffle data
    index = [i for i in range(fdata.shape[0])]
    np.random.seed(seed)
    np.random.shuffle(index)
    
    fdata, ddata, t1, labels, excel = (
        fdata[index],
        ddata[index],
        t1[index],
        labels[index],
        excel[index],
    )
    return fdata, ddata, t1, labels, excel