import copy

import numpy as np

import createDatePlot
from Softmax import*
from Layer import*
from Normalize import*
import createDatePlot as cp
import Classify
import Precise_loss_function
import Get_final_layer_preAct_demands

#第一层：输入为2，输出为3;第二层输入为3，输出为4.......
NETWORK_SHAPE = [2,3,4,5,2]

class Network():
    def __init__(self,network_shape):
        self.shape = network_shape
        self.layers = []
        for i in range(len(network_shape)-1):
            layer = Layer(network_shape[i],network_shape[i+1])
            self.layers.append(layer)

    def network_forward(self,inputs):
        outputs = [inputs]
        for i in range(len(self.layers)):
            layer_sum = self.layers[i].layer_forward(outputs[i])
            if i < len(self.layers)-1:
                layer_output = ReLU(layer_sum)
                layer_output = normalize(layer_output)
            else:
                layer_output = Softmax(layer_sum)
            outputs.append(layer_output)
        print(outputs)
        return outputs






if __name__ == '__main__':
    # network = Network(NETWORK_SHAPE)
    # # print(network.layers[0].weights)
    # # print(network.shape)
    # #print(len(network.layers))
    # inputs = np.array([[1, 2],
    #                    [4, 5],
    #                    [6, 7]])
    # print(network.network_forward(inputs))
    data = cp.creat_data(5)
    cp.plot_data(data,"Right classification")
    #print(data)
    inputs = data[:,(0,1)]
    targets = copy.deepcopy(data[:,2])#标准答案

    #print(inputs)
    network = Network(NETWORK_SHAPE)
    outputs = network.network_forward(inputs)
    classification = Classify.classify(outputs[-1])
    print(classification)
    data[:,2] = classification
    print(data)

    loss = Precise_loss_function.prescise_loss_function(outputs[-1],targets)
    print(loss)
    demands = Get_final_layer_preAct_demands.get_final_layer_preAct_demands(outputs[-1],targets)
    print(demands)
    cp.plot_data(data,"Before training")