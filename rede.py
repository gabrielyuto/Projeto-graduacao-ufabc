import numpy as np

def gennet_inh_lat(input_neuron, compete_neuron):
  m11 = np.zeros((input_neuron, input_neuron))
  m12 = np.zeros((input_neuron, compete_neuron))

  m21 = 0.1 * np.ones((compete_neuron, input_neuron)) + 0.1 * np.random.rand(compete_neuron, input_neuron)
  m22 = -(0.8 * (np.ones((compete_neuron, compete_neuron)) - np.eye(compete_neuron))) + 0.6 * np.eye(compete_neuron)

  w = np.block([[m11, m12], [m21, m22]])

  m22 = np.zeros_like(m22)
  w_mask = np.block([[m11, m12], [m21, m22]])
  mask = w_mask > np.zeros_like(w_mask)

  return w, mask

def activation_function(activation, shift):
  return 1 / (1 + np.exp(-70 * (activation - shift)))


def increment_w(output, output_antes, mask, w):
    learning_factor = 0.001
    term_a = np.dot(output, output_antes.T)
    term_b = np.ones_like(output) * output_antes.T * w

    return (learning_factor * (term_a - term_b)) * mask


def calculation_shift(output, shift):
    displacement_speed = 0.025

    return (displacement_speed * output + shift) / (1 + displacement_speed)


def neural_network(w, mask, train_data):
    number_neurons = w.shape[0]
    shift = 0.5 * np.ones((number_neurons, 1))
    epochs = 1000
    incw = np.zeros_like(w)
    number_entries, patterns = train_data.shape
    camadas = 1

    for i in range(epochs):
        for j in range(patterns):
            output = np.zeros((number_neurons, 1))
            output_antes = output

            PAT = train_data[:, j]
            output[0:number_entries, 0] = PAT

            for k in range(camadas):
                w = w + incw
                inet = np.dot(w, output)

                output = activation_function(inet, shift)

                output = (inet > 0.0) * output
                output[0:number_entries, 0] = PAT

                incw = increment_w(output, output_antes, mask, w)
                shift = calculation_shift(output, shift)
                output_antes = output

    return w, shift

def teste(w, input, shift):
  number_neurons = w.shape[0]
  new_input = np.zeros((number_neurons, 1))
  number_entries = input.shape[0]

  new_input[0:number_entries, 0] = input

  inet = np.dot(w, new_input)
  results = np.array([])

  for i in range(len(inet)):
    result = activation_function(inet[i], shift[i])
    results = np.append(results, result[0])

  return (inet > 0.0).flatten() * results
