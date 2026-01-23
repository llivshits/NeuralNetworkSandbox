import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine  # UCI Wine dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
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
    encoded = np.zeros((len(y), 3))
    for i, label in enumerate(y):
        encoded[i][label] = 1
    return encoded


y_train_encoded = one_hot_encode(y_train)
y_test_encoded = one_hot_encode(y_test)


input_neurons = [InputNeuron() for _ in range(13)]
hidden_neurons = [HiddenNeuron(bias=0.0) for _ in range(8)]
output_neurons = [OutputNeuron(bias=0.0) for _ in range(3)]

for i_neuron in input_neurons:
    for h_neuron in hidden_neurons:
        edge = Edge(val=np.random.randn() * 0.01, fan_in=13)
        i_neuron.outputs.append(edge)
        h_neuron.inputs.append(edge)

for h_neuron in hidden_neurons:
    for o_neuron in output_neurons:
        edge = Edge(val=0, fan_in=8)
        h_neuron.outputs.append(edge)
        o_neuron.inputs.append(edge)

network = NeuralNetwork(input_neurons, hidden_neurons, output_neurons)

losses = network.train(
    x_train=x_train.tolist(),
    y_train=y_train_encoded.tolist(),
    epochs=100,
    learning_rate=0.01,
)

# Plot and Confusion Matrtix
plt.figure(figsize=(6, 4))
plt.plot(range(len(losses)), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Training loss")
plt.tight_layout()
plt.show()

preds = []
trues = []
for x, target in zip(x_test, y_test_encoded):
    prediction = network.forward(x.tolist())
    preds.append(np.argmax(prediction))
    trues.append(np.argmax(target))

accuracy = sum(int(p == t) for p, t in zip(preds, trues)) / len(trues)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(trues, preds, target_names=wine.target_names))

cm = confusion_matrix(trues, preds)
ConfusionMatrixDisplay(cm, display_labels=wine.target_names).plot(cmap="Blues")
plt.title("Wine NN confusion matrix")
plt.tight_layout()
plt.show()
