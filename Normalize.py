import numpy as np

def normalize(array):
    max_number = np.max(np.absolute(array),axis=1,keepdims=True)
    scale_rate = np.where(max_number == 0,0,1/max_number)
    norm = array*scale_rate
    return norm