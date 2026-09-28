import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def encode_features(features):
    circuit = QuantumCircuit(len(features), len(features))

    for i, feature in enumerate(features):
        angle = np.pi * feature
        circuit.ry(angle, i)

    circuit.measure(range(len(features)), range(len(features)))

    return circuit


def main():
    # Example classical features
    features = [0.25, 0.75]

    circuit = encode_features(features)

    print("Input features:")
    print(features)

    print("\nQuantum circuit:")
    print(circuit)

    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()

    counts = result.get_counts()

    print("\nMeasurement results:")
    print(counts)


if __name__ == "__main__":
    main()