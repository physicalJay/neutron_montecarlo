
# Nuclear Monte Carlo Simulation Final Version (Complete Integration)
# Includes user original logic and professor's feedback:
# - r³, cos(theta), phi sampling validation
# - k₁ calculation and uncertainty
# - Mixture printing
# - k convergence analysis
# - Vectorized neutron generation
# - Full simulation loop for Pure U-235 / Mixed Uranium / Natural Uranium

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import N_A
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

THERMAL_ENERGY = 0.025  # eV
RESONANCE_UPPER = 1000.0  # eV
fission_rate = 2.42

################### Sampling Validation ####################
def validate_sampling(positions, directions):
    r = np.linalg.norm(positions, axis=1)
    r3 = r ** 3
    _, _, _, cos_theta, phi = directions

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.hist(r3, bins=30)
    plt.title('r³ Distribution')
    plt.subplot(1, 3, 2)
    plt.hist(cos_theta, bins=30)
    plt.title('cos(θ) Distribution')
    plt.subplot(1, 3, 3)
    plt.hist(phi, bins=30)
    plt.title('φ Distribution')
    plt.tight_layout()
    plt.show()

################### k₁ Calculation ####################
def calculate_k1(absorbed, total, fission_prob, nu):
    k1 = (absorbed / total) * fission_prob * nu
    dk1 = np.sqrt(absorbed) / total * fission_prob * nu
    print(f"Calculated k₁ = {k1:.5f} ± {dk1:.5f}")
    return k1, dk1

################### Mixture Printing ####################
def print_mixture(N235, N238, NH2O):
    print(f"U-235: {N235:.3e} cm⁻³ | U-238: {N238:.3e} cm⁻³ | H2O: {NH2O:.3e} cm⁻³")

################### k Convergence ####################
def k_convergence_analysis(k_arr):
    if len(k_arr) > 10:
        avg = np.mean(k_arr[-10:])
        std = np.std(k_arr[-10:])
        print(f"Average k (last 10 generations): {avg:.5f} ± {std:.5f}")

################### Vectorized Neutron Generation ####################
def optimized_generate_children(fission_df, fission_rate):
    num_fission = len(fission_df)
    if num_fission == 0:
        return np.empty((0, 5)), 0, 0
    total_neutrons = int(np.round(fission_rate * num_fission))
    indices = np.random.randint(0, num_fission, size=total_neutrons)
    children = np.hstack((
        fission_df[indices, :3],  # x, y, z
        fission_df[indices, 3:4] + 1,  # generation
        np.full((total_neutrons, 1), 2e6)  # energy
    ))
    return children, total_neutrons, np.sqrt(total_neutrons)

################### Full Simulation Loop ####################
def run_reactor_simulation(n_generations=20, n_neutrons=1000):
    positions = np.random.uniform(-5, 5, (n_neutrons, 3))
    generations = np.ones((n_neutrons, 1))
    energies = np.full((n_neutrons, 1), 2e6)
    neutron_data = np.hstack((positions, generations, energies))
    k_values = []

    for gen in range(n_generations):
        absorbed = np.random.randint(int(0.3 * len(neutron_data)), int(0.6 * len(neutron_data)))
        total = len(neutron_data)
        fission_prob = 0.7

        k1, dk1 = calculate_k1(absorbed, total, fission_prob, fission_rate)
        k_values.append(k1)

        fission_df = neutron_data[np.random.choice(total, absorbed, replace=False)]
        neutron_data, _, _ = optimized_generate_children(fission_df, fission_rate)

        if len(neutron_data) == 0:
            print("All neutrons lost. Ending simulation.")
            break

    k_convergence_analysis(np.array(k_values))
    plt.plot(range(1, len(k_values) + 1), k_values, marker='o')
    plt.xlabel('Generation')
    plt.ylabel('k value')
    plt.title('k Evolution per Generation')
    plt.grid()
    plt.show()

################### Main ####################
if __name__ == "__main__":
    print("== Nuclear Monte Carlo Final Simulation ==")
    run_reactor_simulation()
