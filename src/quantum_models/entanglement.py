from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def create_entangled_circuit():
    circuit = QuantumCircuit(2, 2)

    # Create superposition on the first qubit
    circuit.h(0)

    # Entangle the two qubits
    circuit.cx(0, 1)

    # Measure both qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    return circuit


def main():
    circuit = create_entangled_circuit()

    print("Quantum Circuit:")
    print(circuit)

    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()

    counts = result.get_counts()

    print("\nMeasurement results:")
    print(counts)


if __name__ == "__main__":
    main()


