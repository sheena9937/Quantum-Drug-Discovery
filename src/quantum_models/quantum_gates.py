from qiskit import QuantumCircuit


def demonstrate_gates():
    circuit = QuantumCircuit(2)

    # X gate
    circuit.x(0)

    # H gate
    circuit.h(1)

    # CNOT gate
    circuit.cx(0, 1)

    return circuit


def main():
    circuit = demonstrate_gates()

    print("Quantum Gates Circuit:")
    print(circuit)


if __name__ == "__main__":
    main()