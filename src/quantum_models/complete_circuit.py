from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def create_circuit():
    circuit = QuantumCircuit(2, 2)

    # Step 1: Put qubit 0 into superposition
    circuit.h(0)

    # Step 2: Entangle qubit 0 and qubit 1
    circuit.cx(0, 1)

    # Step 3: Apply X gate to qubit 1
    circuit.x(1)

    # Step 4: Measure both qubits
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    return circuit


def main():
    circuit = create_circuit()

    print("Complete Quantum Circuit:")
    print(circuit)

    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()

    counts = result.get_counts()

    print("\nMeasurement results:")
    print(counts)


if __name__ == "__main__":
    main()