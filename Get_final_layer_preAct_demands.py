import numpy as np


def get_final_layer_preAct_demands(predicted_valued,target_vector):
    target = np.zeros((len(target_vector),2))
    target[:,1] = target_vector
    target[:,0] = 1 - target_vector

    for i in range(len(target_vector)):
        if np.dot(target[i],predicted_valued[i]) > 0.5:
            target[i] = np.array([0,0])
        else:
            target[i] = (target[i] - 0.5)*2
        return target


if __name__ == '__main__':
    real = np.array([1,0,1,0,1])
    predicted = np.array([[1,2],
                          [2,2],
                          [3,4],
                          [3,2],
                          [0,1]]
                         )
    print(get_final_layer_preAct_demands(predicted,real))