import numpy as np
from sklearn.datasets import load_wine  # UCI Wine dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neuralnetwork import InputNeuron, HiddenNeuron, OutputNeuron, NeuralNetwork, Edge

wine = load_wine()
x = wine.data
y = wine.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)


def one_hot_encode(y):
    encoded = np.zeros(len(y), 3)
    for i, label in enumerate(y):
        encoded[i][label] = 1
    return encoded


y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)
x_train_encoded = one_hot_encode(x_train)
x_test_encoded = one_hot_encode(x_test)

input_neurons = [InputNeuron() for _ in range(13)]
hidden_neurons = [HiddenNeuron(bias=0.0) for _ in range(8)]
output_neurons = [OutputNeuron(bias=0.0) for _ in range(3)]

for i_neuron in input_neurons:
    for h_neuron in hidden_neurons:
        edge = Edge(val=0, fan_in=13, fan_out=8)
        i_neuron.outputs.append(edge)
        h_neuron.inputs.append(edge)

for h_neuron in hidden_neurons:
    for o_neuron in output_neurons:
        edge = Edge(val=0, fan_in=8, fan_out=3)
        h_neuron.outputs.append(edge)
        o_neuron.inputs.append(edge)

network = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)

network.train(
    X_train=x_train_encoded.tolist(),
    y_train=y_train_encoded.tolist(),
    epochs=100,
    learning_rate=0.01,
)

correct = 0
for x, target in zip(x_test, y_test_encoded):
    prediction = nn.forward(x.tolist())
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(target)
    if predicted_class == actual_class:
        correct += 1

accuracy = correct / len(x_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
