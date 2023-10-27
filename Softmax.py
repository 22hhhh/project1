import numpy as np

def Softmax(inputs):
    max_values = np.max(inputs,axis=1,keepdims=True)
    slided_inputs = inputs - max_values
    exp_values = np.exp(slided_inputs)
    norm_base = np.sum(exp_values,axis=1,keepdims=True)
    norm_values = exp_values / norm_base
    # return max_values
    # print(slided_inputs)
    # print(exp_values)
    return norm_values


if __name__ == '__main__':
    inputs = np.array([[1,2],
                       [4,5],
                       [6,7]])
    print(Softmax(inputs))