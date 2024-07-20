import rede
import numpy as np

if __name__ == '__main__':
    input_neuron = 5
    compete_neuron = 3

    w, mask = rede.gennet_inh_lat(input_neuron, compete_neuron)

    # Padrões: A, B, A, C, A
    P = np.array([[0.1, 0.2, 0.0, 0.0, 0.7],
                  [0.0, 0.0, 0.4, 0.6, 0.8],
                  [0.3, 0.1, 0.0, 0.0, 0.6],
                  [0.8, 0.2, 0.0, 0.0, 0.0],
                  [0.2, 0.1, 0.0, 0.0, 0.7]])

    train_data = np.array(P).T
    matrix_weights, shift = rede.neural_network(w, mask, train_data)

    # Teste A
    input_test = np.array([0.1, 0.2, 0.0, 0.0, 0.7]).T

    result = rede.teste(matrix_weights, input_test, shift)

    print(result)