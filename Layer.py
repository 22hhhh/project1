import numpy as np
import Normalize

BATCH_SIZE = 5

def ReLU(inputs):
    #对矩阵每一行进行偏置
    return np.maximum(0,inputs)

class Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.rand(n_inputs,n_neurons)
        self.biases = np.random.rand(n_neurons)

    def layer_forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        #self.output = ReLU(sum1)
        #return self.output
        return self.output

    def layer_backward(self,afterWeights_demands,preWeight_values):
        preWeight_demands = np.dot(afterWeights_demands,self.weights.T)

        condition = (preWeight_values > 0)
        values_derivatives = np.where(condition,1,0)#匿名函数：如condition大于0则等于1，否则等于0。从而完成对激活函数的求导。
        preActs_demands = values_derivatives*preWeight_demands
        norm_preActs_demands = Normalize.normalize(preActs_demands)

        weight_adjust_matrix = self.get_weight_adjust_matrix(preWeight_values,afterWeights_demands)
        norm_weight_adjust_matrix = Normalize.normalize(weight_adjust_matrix)
        return (norm_preActs_demands,norm_weight_adjust_matrix)

    def get_weight_adjust_matrix(self,preWeights_values,aftWeights_demands):
        plain_weights = np.full(self.weights.shape,1)
        weights_adjust_matrix = np.full(self.weights.shape,0)
        plain_weights_T = plain_weights.T

        for i in range(BATCH_SIZE):
            weights_adjust_matrix += (plain_weights_T*preWeights_values[i,:].T*aftWeights_demands[i,:])
        weights_adjust_matrix = weights_adjust_matrix/BATCH_SIZE
        return weights_adjust_matrix

if __name__ == '__main__':

    inputs = np.array([[1,2],
                       [4,5],
                       [6,7]])
    layer1 = Layer(2,3)
    layer2 = Layer(3,4)
    layer3 = Layer(4,2)

    output1 = layer1.layer_forward(inputs)
    print(output1)