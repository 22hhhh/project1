import numpy as np

def prescise_loss_function(predicted,real):
    real_matrix = np.zeros((len(real),2))
    print(real)
    real_matrix[:,1] = real
    real_matrix[:,0] = 1-real
    print(real_matrix)
    product = np.sum(predicted*real_matrix,axis=1)
    return 1-product