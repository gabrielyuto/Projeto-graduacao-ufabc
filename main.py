import rede
import numpy as np
import pandas as pd
from sklearn import datasets

if __name__ == '__main__':
  iris = datasets.load_iris()
  input_neuron = 5
  compete_neuron = 3

  w, mask = rede.gennet_inh_lat(input_neuron, compete_neuron)

  df = pd.DataFrame(iris.data, columns=iris.feature_names)
  padroes = np.array([df['sepal length (cm)'], df['sepal width (cm)'], df['petal length (cm)'], df['petal width (cm)']])
  metade = padroes.shape[1] // 2
  train_data = padroes[:, :metade]
  # PADRÕES: | A | B | A | C | A |
  # P = np.array([[0.1, 0.2, 0.0, 0.0, 0.7],
  #               [0.0, 0.0, 0.4, 0.6, 0.8],
  #               [0.3, 0.1, 0.0, 0.0, 0.6],
  #               [0.8, 0.2, 0.0, 0.0, 0.0],
  #               [0.2, 0.1, 0.0, 0.0, 0.7]]).T

  # train_data = np.array(P)
  matrix_weights, shift = rede.neural_network(w, mask, train_data)

  # TESTE: | A |
  input_test = np.array([0.1, 0.2, 0.0, 0.0, 0.7]).T

  result = rede.teste(matrix_weights, input_test, shift)

  print(result)
