from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def create_superposition():
    circuit = QuantumCircuit(1, 1)

    # Create superposition
    circuit.h(0)

    # Measure qubit
    circuit.measure(0, 0)

    return circuit


def main():
    circuit = create_superposition()

    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()

    counts = result.get_counts()

    print("Measurement results:")
    print(counts)


if __name__ == "__main__":
    main()
