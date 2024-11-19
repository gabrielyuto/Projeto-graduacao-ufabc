import numpy as np
import bio_neural_network
import pandas as pd

if __name__ == '__main__':
  input_neuron = 5
  compete_neuron = 3

  w, mask = bio_neural_network.gennet_inh_lat(input_neuron, compete_neuron)

  patterns = pd.read_csv('../dados/train.csv')
  patterns = patterns.drop('tipo', axis=1).to_numpy().T

  act = bio_neural_network.neural_network(w, mask, patterns)
  df = pd.DataFrame(act)

  df


  # matrix_weights, shift = bio_neural_network.neural_network(w, mask, patterns)

  # TESTE: | A |
  # input_test = np.array([0.8, 0.2, 0.0, 0.0, 0.0]).T
  #
  # result = bio_neural_network.teste(matrix_weights, input_test, shift)
  #
  # print(result)
