# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:10:58 2024

@author: Mason

"""

import torch
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error
from tqdm import tqdm

"""
Configure experiment families
----------------------------------
Each experiment corresponds to:
 - a circuit depth
 - noise strength
 - number of shots
"""

NUM_QUBITS = 3
DEPTH_RANGE = range(1, 80)           
NOISE_LEVELS = np.linspace(0.0, 0.2, 20)
SHOTS = 2048                         
NUM_OUTCOMES = 2 ** NUM_QUBITS

def build_random_circuit(num_qubits, depth):
    qc = QuantumCircuit(num_qubits)

    for _ in range(depth):
        q = np.random.randint(0, num_qubits)
        if np.random.rand() < 0.5:
            qc.h(q)
        else:
            qc.s(q)

        control, target = np.random.choice(num_qubits, size=2, replace=False)
        qc.cx(control, target)

    qc.measure_all()
    return qc

def build_noise_model(noise_strength):
    noise = NoiseModel()
    dep1 = depolarizing_error(noise_strength, 1)
    amp1 = amplitude_damping_error(noise_strength)
    dep2 = depolarizing_error(noise_strength, 2)

    single_qubit_gates = ['h', 's']
    two_qubit_gates    = ['cx']

    for g in single_qubit_gates:
        noise.add_all_qubit_quantum_error(dep1, [g])
        noise.add_all_qubit_quantum_error(amp1, [g])

    for g in two_qubit_gates:
        noise.add_all_qubit_quantum_error(dep2, [g])

    return noise


def get_distribution(counts):
    probs = np.zeros(NUM_OUTCOMES)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count
    probs = probs / np.sum(probs)
    return probs


def generate_dataset():
    dataset = []
    simulator = AerSimulator()
    
    sequence_length = 4
    
    for depth in tqdm(DEPTH_RANGE, desc="Depth sweep"):
        for eps in NOISE_LEVELS:
            sequence = []
            
            for _ in range(sequence_length):
                qc = build_random_circuit(NUM_QUBITS, depth)
                noise = build_noise_model(eps)
                backend = AerSimulator(noise_model=noise)
                
                compiled = transpile(qc, backend)
                job = backend.run(compiled, shots=SHOTS)
                
                result = job.result()
                counts = result.get_counts()
                
                dist = get_distribution(counts)
                
                conditioning = np.array([
                    depth/max(DEPTH_RANGE),
                    eps/NOISE_LEVELS.max()
                ])
                
                sample = np.concatenate([conditioning, dist])
                sequence.append(sample)
            
            dataset.append(np.array(sequence))
    
    dataset = np.array(dataset)
    tensor = torch.tensor(dataset, dtype=torch.float32)
    
    print(f"\nFinal tensor shape: {tensor.shape}")
    torch.save(tensor, "data/q_dist_data_tensor")
    print(f"Saved tensor with shape: {tensor.shape}")

if __name__ == "__main__":
    generate_dataset()




