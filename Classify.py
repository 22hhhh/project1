import numpy as np

def classify(probabilities):
    classification = np.rint(probabilities[:,1])#取第二列
    return classification
