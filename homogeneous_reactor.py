# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 15:28:02 2025

@author: Gao and Lee

Units:
- Cross sections: barns (1 barn = 1e-24 cm²)
- Number density: cm⁻³
- Macroscopic cross sections: cm⁻¹
- Energy: eV
- Distance: cm
"""

import sys
import os
import matplotlib
# Set matplotlib to use inline backend
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from matplotlib import get_backend
import numpy as np
import numpy.random as rand
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import logging

# Configure plotting settings
plt.rcParams.update({
    'figure.dpi': 100,
    'figure.autolayout': True
})

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Clear all existing plots at the start
plt.close('all')

# Constants for energy thresholds
THERMAL_ENERGY = 0.025  # eV, threshold energy for thermal neutrons
RESONANCE_LOWER = 1.0   # eV, lower bound for resonance region
RESONANCE_UPPER = 1000.0  # eV, upper bound for resonance region (1 keV)

# Create DataFrame from table 10.2
columns = ["Nucleus", "Density (g/cm³)", "σ_f", "σ_c", "σ_a", "σ_s"]

# Fast neutron cross sections
df_U = pd.DataFrame(np.array([
                                ["235U", 18.7, 1, 0.09, 1.09, 4], 
                                ["238U", 18.9, 0.3, 0.07, 0.37, 5],
                                ["natU", 18.9, 4.17, 3.43, 7.60, 8.3]
                                ], dtype=object),
                    columns=columns).set_index("Nucleus")
df_U["σ_T"] = df_U["σ_a"] + df_U["σ_s"]
print("Fast neutron cross sections:")
print(df_U)

# Thermal neutron cross sections
df_U_thermal = pd.DataFrame(np.array([
                            ["235U", 18.7, 579, 101, 680, 10], 
                            ["238U", 18.9, 0, 2.72, 2.72, 8.3],
                            ["natU", 18.9, 4.17, 3.43, 7.60, 8.3]
                            ], dtype=object),
                columns=columns).set_index("Nucleus")
df_U_thermal["σ_T"] = df_U_thermal["σ_a"] + df_U_thermal["σ_s"]
print("\nThermal neutron cross sections:")
print(df_U_thermal)

# Resonance region cross sections (approximate values based on typical behavior)
df_U_resonance = pd.DataFrame(np.array([
                            ["235U", 18.7, 50, 150, 200, 8], 
                            ["238U", 18.9, 0.1, 50, 50.1, 7],
                            ["natU", 18.9, 4.0, 20, 24, 8]
                            ], dtype=object),
                columns=columns).set_index("Nucleus")
df_U_resonance["σ_T"] = df_U_resonance["σ_a"] + df_U_resonance["σ_s"]
print("\nResonance region cross sections (approximate):")
print(df_U_resonance)

#create dataframe from table 10.3
df_moderator = pd.DataFrame(np.array([
                                        ["H2O", 18.01, 1.0, 49.2, 0.66],
                                        ["D2O", 20.02, 1.1, 10.6, 0.001],
                                        ["Graphite", 12.01, 1.6, 4.7, 0.0045]
                                        ], dtype=object), 
                            columns=["Material", "M.Wt (g/mol)", "Density (g/cm³)", "σ_s", "σ_a"]).set_index("Material")
df_moderator["σ_T"] = df_moderator["σ_a"] + df_moderator["σ_s"]
print("\nModerator data:")
print(df_moderator)

# Create thermal moderator data (same as fast for now, but can be modified if needed)
df_moderator_thermal = df_moderator.copy()
df_moderator_resonance = df_moderator.copy()

# constants
from scipy.constants import N_A
U235_MolarMass = 235
U238_MolarMass = 238
fission_rate = 2.42

def sphere_position(r,n):
    # create a sample that each column represents x,y,z, randomly for each row
    # in order to create around n number of neutrons in sphere, the cube need to have n/pi*6 lines
    cube_sample = rand.uniform(-r,r,size=(int(n/np.pi*6+1),3))
    
    in_sphere = np.where(cube_sample[:,0]**2 + 
                         cube_sample[:,1]**2 + 
                         cube_sample[:,2]**2 <= r**2)
    sphere_sample = cube_sample[in_sphere]
    
    return sphere_sample

def cylinder_position(r,l,n):
    cuboid_sample = rand.uniform(np.array([-r,-r,-l/2]),
                                 np.array([r,r,l/2]),
                                 size = (int(n/np.pi*4+1),3))
    in_cylinder = np.where(cuboid_sample[:,0]**2 + 
                           cuboid_sample[:,1]**2 <= r**2)
    cylinder_sample = cuboid_sample[in_cylinder]
    
    return cylinder_sample

def random_direction(n):
    """
    Generate random direction vectors using spherical coordinates
    
    Parameters:
    -----------
    n : int
        Number of vectors to generate
    
    Returns:
    --------
    tuple
        (x, y, z, cos_theta, phi) components of unit vectors and angles
    """
    phi = rand.uniform(0, 2*np.pi, n)
    cos_theta = rand.uniform(-1, 1, n)
    theta = np.arccos(cos_theta)
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return x, y, z, cos_theta, phi

def random_distance(l,n):
    #create random distance from exponential distribution
    return rand.exponential(scale = l,size = n)

def random_1_step(last_position, mean_free_path):
    """
    Move neutrons one step in random directions
    
    Parameters:
    -----------
    last_position : numpy.ndarray
        Array of current positions
    mean_free_path : float
        Mean free path (cm)
    
    Returns:
    --------
    numpy.ndarray
        Array of new positions
    """
    size = len(last_position)
    new_position = np.array([last_position], dtype=float)[0]

    x, y, z, cos_theta, phi = random_direction(size)  # Now unpacking all 5 values
    distances = random_distance(mean_free_path, size)
    x_displacements = x * distances
    y_displacements = y * distances
    z_displacements = z * distances
        
    # add the displacements to the corresponding columns for x,y,z
    new_position[:,0] += x_displacements
    new_position[:,1] += y_displacements
    new_position[:,2] += z_displacements        
    
    return new_position

# Updated elastic scattering energy calculation
def calculate_energy_after_scatter(E_before, A):
    """
    Calculate neutron energy after elastic scattering
    
    Parameters:
    -----------
    E_before : float
        Neutron energy before collision (eV)
    A : float
        Mass number of target nucleus (H=1, C=12, U-235=235, etc.)
    
    Returns:
    --------
    float
        Neutron energy after collision (eV)
    """
    # Random scattering angle cosine (-1 to 1)
    cos_theta = rand.uniform(-1, 1)
    
    # E′=E×(A+1)²/((A−1)²+cos²θ) formula
    E_after = E_before * ((A+1)**2)/((A-1)**2 + cos_theta**2)
    
    return E_after

def determine_energy_region(energy):
    """
    Determine which energy region the neutron belongs to
    
    Parameters:
    -----------
    energy : float
        Neutron energy (eV)
    
    Returns:
    --------
    str
        'thermal', 'resonance', or 'fast'
    """
    if energy <= THERMAL_ENERGY:
        return 'thermal'
    elif energy <= RESONANCE_UPPER:
        return 'resonance'
    else:
        return 'fast'

def test_energy(n,A):
    x = np.arange(0, n)
    E = np.zeros(n)
    E[0] = 2e6  # Start with 2 MeV (2,000,000 eV)
    for i in range(1,n):
        E[i] = calculate_energy_after_scatter(E[i-1], A)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, E, '-o')
    plt.yscale('log')
    plt.axhline(y=THERMAL_ENERGY, color='r', linestyle='--', label=f'Thermal (<{THERMAL_ENERGY} eV)')
    plt.axhline(y=RESONANCE_UPPER, color='g', linestyle='--', label=f'Resonance (<{RESONANCE_UPPER} eV)')
    plt.title(f'Energy Reduction by Elastic Scattering (A={A})')
    plt.xlabel('Collision Number')
    plt.ylabel('Energy (eV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def reactor_pure235_sphere(mass):
    density = df_U.loc['235U',"Density (g/cm³)"]
    r = (mass/density*3/4/np.pi)**(1/3)
    n = 1000  # Changed from 10000 to 1000
    initial_position = sphere_position(r, n)
    coeff = np.array([r])
    
    # Calculate number density
    N235 = density * N_A/U235_MolarMass
    N = np.array([N235, 0, 0])
    
    # Fast neutron cross sections
    macro_sigma_total_fast = df_U.loc["235U","σ_T"]*N235*1e-24
    mf_path_fast = 1/macro_sigma_total_fast
    
    # Thermal neutron cross sections
    macro_sigma_total_thermal = df_U_thermal.loc["235U","σ_T"]*N235*1e-24
    mf_path_thermal = 1/macro_sigma_total_thermal
    
    # Resonance region cross sections
    macro_sigma_total_resonance = df_U_resonance.loc["235U","σ_T"]*N235*1e-24
    mf_path_resonance = 1/macro_sigma_total_resonance
    
    shape = 'sphere'
    return initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance

def reactor_setup_noInput(shape, U_info, moderator_info): # U_info, moderator_info are dataframe, with column name type and first row fraction
    
    if shape == "sphere":
        r = 10
        n = 3000
        initial_position = sphere_position(r, n)
        coeff = np.array([r])
    elif shape == "cylinder":
        r = 50
        l = 100
        n = 10000
        initial_position = cylinder_position(r, l, n)
        coeff = np.array([r,l]) # give the information about the cylinder
    else:
        print("invalid shape")
        return "error"
    
    #get info
    f_U235 = U_info.loc[0,'235U']
    f_U238 = U_info.loc[0,'238U']
    f_H2O = moderator_info.loc[0,'H2O']
    
    # suppose f is mass density ratio
    N235 = f_U235 * 18.7* N_A/U235_MolarMass
    N238 = f_U238 * 18.9* N_A/U238_MolarMass
    NH2O = f_H2O * 1* N_A/18.01
    N = np.array([N235,N238,NH2O])
    
    # Fast neutron cross sections
    sigma_total_fast = np.array([df_U.loc["235U","σ_T"],df_U.loc["238U","σ_T"],df_moderator.loc["H2O","σ_T"]])*1e-24
    macro_sigma_total_fast = np.sum(N*sigma_total_fast)
    mf_path_fast = 1/macro_sigma_total_fast
    
    # Thermal neutron cross sections
    sigma_total_thermal = np.array([df_U_thermal.loc["235U","σ_T"],df_U_thermal.loc["238U","σ_T"],df_moderator_thermal.loc["H2O","σ_T"]])*1e-24
    macro_sigma_total_thermal = np.sum(N*sigma_total_thermal)
    mf_path_thermal = 1/macro_sigma_total_thermal
    
    # Resonance region cross sections
    sigma_total_resonance = np.array([df_U_resonance.loc["235U","σ_T"],df_U_resonance.loc["238U","σ_T"],df_moderator_resonance.loc["H2O","σ_T"]])*1e-24
    macro_sigma_total_resonance = np.sum(N*sigma_total_resonance)
    mf_path_resonance = 1/macro_sigma_total_resonance
    
    return initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance

def whether_outside(new_position,shape,coeff):
    if shape == "sphere":
        condition = (new_position[:,0]**2 + 
                     new_position[:,1]**2 +
                     new_position[:,2]**2) > coeff[0]**2
        return condition,np.all(condition)
    
    elif shape == "cylinder":
        condition = ((new_position[:,0]**2 + new_position[:,1]**2) > coeff[0]**2) | (np.abs(new_position[:,2])>(coeff[1]/2))
        return condition,np.all(condition)

# Updated collision_result function with three energy regions
def collision_result(df, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance):
    """
    Determine neutron collision results (scattering, capture, fission)
    Apply different cross sections based on energy region
    """
    # Convert DataFrame to numpy arrays for faster processing
    energy = df['energy'].values
    energy_region = df['energy_region'].values
    status = df['status'].values
    
    # Get cross sections for each energy region
    fast_mask = energy_region == 'fast'
    thermal_mask = energy_region == 'thermal'
    resonance_mask = energy_region == 'resonance'
    
    # Calculate reaction probabilities for each region
    sigma_fission = np.array([df_U.loc["235U","σ_f"], df_U.loc["238U","σ_f"], 0])*1e-24
    sigma_scatter = np.array([df_U.loc["235U","σ_s"], df_U.loc["238U","σ_s"], df_moderator.loc["Graphite","σ_s"]])*1e-24
    sigma_capture = np.array([df_U.loc["235U","σ_c"], df_U.loc["238U","σ_c"], df_moderator.loc["Graphite","σ_a"]])*1e-24
    
    # Calculate macroscopic cross sections for each region
    macro_sigma_fission = np.sum(N * sigma_fission)
    macro_sigma_scatter = np.sum(N * sigma_scatter)
    macro_sigma_capture = np.sum(N * sigma_capture)
    
    # Calculate probabilities
    probability_fission = macro_sigma_fission / np.array([
        macro_sigma_total_fast,
        macro_sigma_total_thermal,
        macro_sigma_total_resonance
    ])
    probability_capture = macro_sigma_capture / np.array([
        macro_sigma_total_fast,
        macro_sigma_total_thermal,
        macro_sigma_total_resonance
    ]) + probability_fission
    
    # Generate random numbers for all neutrons at once
    dice = rand.uniform(0, 1, len(df))
    
    # Determine collision outcomes
    status[fast_mask] = np.where(dice[fast_mask] <= probability_fission[0], 'fission',
                                np.where(dice[fast_mask] <= probability_capture[0], 'capture', 'scatter'))
    status[thermal_mask] = np.where(dice[thermal_mask] <= probability_fission[1], 'fission',
                                   np.where(dice[thermal_mask] <= probability_capture[1], 'capture', 'scatter'))
    status[resonance_mask] = np.where(dice[resonance_mask] <= probability_fission[2], 'fission',
                                     np.where(dice[resonance_mask] <= probability_capture[2], 'capture', 'scatter'))
    
    # Handle scattering
    scatter_mask = status == 'scatter'
    if np.any(scatter_mask):
        # Determine collision nucleus for scattered neutrons
        nuclei_probabilities = N * sigma_scatter / macro_sigma_scatter
        nuclei_cumulative_prob = np.cumsum(nuclei_probabilities)
        
        dice_nuclei = rand.uniform(0, 1, np.sum(scatter_mask))
        nuclei_indices = np.searchsorted(nuclei_cumulative_prob, dice_nuclei)
        A_values = np.array([235, 238, 12])[nuclei_indices]
        
        # Calculate energy after scattering
        old_energy = energy[scatter_mask]
        cos_theta = rand.uniform(-1, 1, np.sum(scatter_mask))
        energy[scatter_mask] = old_energy * ((A_values+1)**2)/((A_values-1)**2 + cos_theta**2)
        
        # Update energy region
        energy_region[scatter_mask] = np.where(energy[scatter_mask] <= THERMAL_ENERGY, 'thermal',
                                             np.where(energy[scatter_mask] <= RESONANCE_UPPER, 'resonance', 'fast'))
    
    # Update DataFrame
    df['status'] = status
    df['energy'] = energy
    df['energy_region'] = energy_region

def test_collision(n, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance):
    df = pd.DataFrame({
        'status': [np.nan] * n,
        'energy': [THERMAL_ENERGY] * n,  # Start with thermal energy
        'energy_region': ['thermal'] * n
    })

    collision_result(df, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance)
    # Count occurrences of each event type
    event_counts = Counter(df['status'])

    # Extract labels and counts
    labels = list(event_counts.keys())
    counts = list(event_counts.values())

    # Calculate total number of events
    total_events = sum(counts)

    # Convert counts to percentages
    percentages = [(count / total_events) * 100 for count in counts]

    # Create the bar chart with percentage on y-axis
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, percentages, color=['red', 'blue', 'green'])

    # Add text labels on top of bars
    for bar, percentage, count in zip(bars, percentages, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f"{percentage:.1f}%\n({count})", 
                 ha='center', va='bottom', fontsize=10)

    # Add labels and title
    plt.xlabel("Event Type")
    plt.ylabel("Percentage (%)")
    plt.title("Percentage Distribution of Nuclear Events")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show the plot
    plt.show()    

# Updated process function to handle three energy regions
def process(df_neutron_walking, mean_free_path_fast, mean_free_path_thermal, mean_free_path_resonance, 
           shape, coeff, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance):
    """
    Process neutron movement and collisions for one generation
    """
    step_taken = 0
    all_outside = False
    df_neutron_NoWalking = pd.DataFrame()
    
    # Convert DataFrame to numpy arrays for faster processing
    position = df_neutron_walking.iloc[:,0:3].values
    energy_region = df_neutron_walking['energy_region'].values
    status = df_neutron_walking['status'].values
    
    while len(df_neutron_walking) != 0:
        # Separate neutrons by energy region
        fast_mask = energy_region == 'fast'
        thermal_mask = energy_region == 'thermal'
        resonance_mask = energy_region == 'resonance'
        
        # Process each energy group
        if np.any(fast_mask):
            x, y, z, cos_theta, phi = random_direction(np.sum(fast_mask))
            distances = random_distance(mean_free_path_fast, np.sum(fast_mask))
            position[fast_mask, 0] += x * distances
            position[fast_mask, 1] += y * distances
            position[fast_mask, 2] += z * distances
        
        if np.any(thermal_mask):
            x, y, z, cos_theta, phi = random_direction(np.sum(thermal_mask))
            distances = random_distance(mean_free_path_thermal, np.sum(thermal_mask))
            position[thermal_mask, 0] += x * distances
            position[thermal_mask, 1] += y * distances
            position[thermal_mask, 2] += z * distances
        
        if np.any(resonance_mask):
            x, y, z, cos_theta, phi = random_direction(np.sum(resonance_mask))
            distances = random_distance(mean_free_path_resonance, np.sum(resonance_mask))
            position[resonance_mask, 0] += x * distances
            position[resonance_mask, 1] += y * distances
            position[resonance_mask, 2] += z * distances
        
        # Check for neutrons that have left the volume
        if shape == "sphere":
            condition = (position[:,0]**2 + position[:,1]**2 + position[:,2]**2) > coeff[0]**2
        else:  # cylinder
            condition = ((position[:,0]**2 + position[:,1]**2) > coeff[0]**2) | (np.abs(position[:,2]) > (coeff[1]/2))
        
        status = np.where(condition, 'outside', 'inside')
        
        # Add neutrons that left to the non-walking group
        outside_mask = status == 'outside'
        if np.any(outside_mask):
            df_outside = df_neutron_walking[outside_mask].copy()
            df_outside.iloc[:,0:3] = position[outside_mask]
            df_outside['status'] = status[outside_mask]
            df_neutron_NoWalking = pd.concat([df_neutron_NoWalking, df_outside], ignore_index=True)
        
        # Update remaining neutrons
        inside_mask = status == 'inside'
        if not np.any(inside_mask):
            break
            
        df_neutron_walking = df_neutron_walking[inside_mask].copy()
        df_neutron_walking.iloc[:,0:3] = position[inside_mask]
        df_neutron_walking['status'] = status[inside_mask]
        energy_region = energy_region[inside_mask]
        
        # Determine collision outcomes
        collision_result(df_neutron_walking, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance)
        
        # Add neutrons that were absorbed to the non-walking group
        absorbed_mask = df_neutron_walking['status'].isin(['fission','capture'])
        if np.any(absorbed_mask):
            df_absorbed = df_neutron_walking[absorbed_mask].copy()
            df_neutron_NoWalking = pd.concat([df_neutron_NoWalking, df_absorbed], ignore_index=True)
        
        # Update remaining neutrons
        scatter_mask = df_neutron_walking['status'] == 'scatter'
        if not np.any(scatter_mask):
            break
            
        df_neutron_walking = df_neutron_walking[scatter_mask].copy()
        position = position[inside_mask][scatter_mask]
        energy_region = df_neutron_walking['energy_region'].values
        status = df_neutron_walking['status'].values
        
        step_taken += 1
        logger.debug(f"Step {step_taken}: {len(df_neutron_walking)} neutrons remaining")
    
    logger.info(f"Generation completed in {step_taken} steps")
    return position, step_taken, df_neutron_NoWalking

# Modified to generate new neutrons with fast energy (fission neutrons)
def generate_children_neutron(df_neutron_last_generation, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance):
    '''
    Parameters
    ----------
    df_neutron_last_generation : pandas dataframe
        Neutrons from the previous generation.
    N : numpy.ndarray
        Array of number densities for each material
    macro_sigma_total_fast : float
        Total macroscopic cross section for fast neutrons
    macro_sigma_total_thermal : float
        Total macroscopic cross section for thermal neutrons
    macro_sigma_total_resonance : float
        Total macroscopic cross section for resonance region neutrons

    Returns
    -------
    df_new_generation : pandas dataframe
        New generation of neutrons.
    k : float
        Multiplication factor.
    dk : float
        Error in multiplication factor.
    '''
    # Safety check: if no fission events, return empty dataframe and k=0
    if len(df_neutron_last_generation) == 0 or len(df_neutron_last_generation[df_neutron_last_generation["status"] == 'fission']) == 0:
        empty_df = pd.DataFrame(columns=["x","y",'z','status','generation','energy','energy_region'])
        return empty_df, 0, 0
    
    # Get fission cross sections based on the energy region of the neutron
    sigma_fission = np.array([df_U.loc["235U","σ_f"],df_U.loc["238U","σ_f"],0])*1e-24
    macro_sigma_fission = np.sum(N * sigma_fission)
    fission_p = macro_sigma_fission/macro_sigma_total_fast
    
    fission_df = df_neutron_last_generation[df_neutron_last_generation["status"] == 'fission']
    k = len(fission_df)/len(df_neutron_last_generation) * fission_rate
    dk = np.sqrt(1/len(fission_df) - 1/len(df_neutron_last_generation))
    
    generation_factor = len(df_neutron_last_generation)/len(fission_df)
    
    # Create new neutrons
    df_new_generation = pd.DataFrame(columns = fission_df.columns)
    for i in range(len(fission_df)):
        position = fission_df.iloc[i,0:3].to_numpy()
        one_child_neutron = pd.DataFrame(columns = fission_df.columns)
        one_child_neutron.loc[0,one_child_neutron.columns[:3]] = position
        one_child_neutron.loc[0,'status'] = "inside"
        one_child_neutron.loc[0,'generation'] = fission_df.iloc[0,fission_df.columns.get_loc('generation')] + 1
        
        one_child_neutron.loc[0,'energy'] = 2e6  # 2 MeV
        one_child_neutron.loc[0,'energy_region'] = 'fast'
        
        n = int(generation_factor)
        if rand.uniform(0,1) < (generation_factor % 1):
            n += 1
        
        children_for_this_fission = pd.DataFrame([one_child_neutron.iloc[0].tolist()] * n, columns = one_child_neutron.columns)
        df_new_generation = pd.concat([df_new_generation, children_for_this_fission], ignore_index=True)
    
    return df_new_generation, k, dk

def reactor_mixed_uranium_sphere(mass_total, u235_fraction, moderator_fraction=600):
    """
    Spherical reactor simulation with mixed U-235, U-238, and moderator (graphite)
    
    Parameters:
    -----------
    mass_total : float
        Total uranium mass (g)
    u235_fraction : float
        Mass fraction of U-235 (between 0 and 1)
    moderator_fraction : float
        Mass ratio of moderator to total uranium (default: 600 for graphite)
    """
    # Input validation
    if mass_total <= 0:
        raise ValueError("Total mass must be positive")
    if not 0 <= u235_fraction <= 1:
        raise ValueError("U235 fraction must be between 0 and 1")
    if moderator_fraction < 0:
        raise ValueError("Moderator fraction must be non-negative")

    # Calculate masses of U-235 and U-238
    mass_u235 = mass_total * u235_fraction
    mass_u238 = mass_total * (1 - u235_fraction)
    mass_moderator = mass_total * moderator_fraction
    
    # Calculate density (weighted average of the mixture)
    density_u235 = df_U.loc['235U', "Density (g/cm³)"]
    density_u238 = df_U.loc['238U', "Density (g/cm³)"]
    density_graphite = df_moderator.loc['Graphite', "Density (g/cm³)"]
    
    # Calculate total volume and density
    volume_u235 = mass_u235 / density_u235
    volume_u238 = mass_u238 / density_u238
    volume_graphite = mass_moderator / density_graphite
    total_volume = volume_u235 + volume_u238 + volume_graphite
    
    # Calculate average density
    density_mix = (mass_total + mass_moderator) / total_volume
    
    # Calculate radius
    r = ((mass_total + mass_moderator)/density_mix*3/4/np.pi)**(1/3)
    n = 1000  # Changed from 10000 to 1000
    initial_position = sphere_position(r, n)
    coeff = np.array([r])
    
    # Calculate atomic number densities
    N235 = (mass_u235/mass_total) * density_mix * N_A/U235_MolarMass
    N238 = (mass_u238/mass_total) * density_mix * N_A/U238_MolarMass
    N_graphite = (mass_moderator/(mass_total + mass_moderator)) * density_mix * N_A/12.01  # 12.01 is molar mass of carbon
    N = np.array([N235, N238, N_graphite])
    
    # Fast neutron cross sections
    macro_sigma_total_fast = (df_U.loc["235U","σ_T"]*N235 + 
                            df_U.loc["238U","σ_T"]*N238 + 
                            df_moderator.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_fast = 1/macro_sigma_total_fast
    
    # Thermal neutron cross sections
    macro_sigma_total_thermal = (df_U_thermal.loc["235U","σ_T"]*N235 + 
                               df_U_thermal.loc["238U","σ_T"]*N238 + 
                               df_moderator_thermal.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_thermal = 1/macro_sigma_total_thermal
    
    # Resonance region cross sections
    macro_sigma_total_resonance = (df_U_resonance.loc["235U","σ_T"]*N235 + 
                                 df_U_resonance.loc["238U","σ_T"]*N238 + 
                                 df_moderator_resonance.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_resonance = 1/macro_sigma_total_resonance
    
    shape = 'sphere'
    return initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance

def reactor_natural_uranium_sphere(mass_total, moderator_fraction=600):
    """
    Spherical reactor simulation with natural uranium and moderator (graphite)
    
    Parameters:
    -----------
    mass_total : float
        Total uranium mass (g)
    moderator_fraction : float
        Mass ratio of moderator to total uranium (default: 600 for graphite)
    """
    mass_moderator = mass_total * moderator_fraction
    
    # Get density values
    density_natU = df_U.loc['natU', "Density (g/cm³)"]
    density_graphite = df_moderator.loc['Graphite', "Density (g/cm³)"]
    
    # Calculate volumes and total density
    volume_natU = mass_total / density_natU
    volume_graphite = mass_moderator / density_graphite
    total_volume = volume_natU + volume_graphite
    
    # Calculate average density
    density_mix = (mass_total + mass_moderator) / total_volume
    
    # Calculate radius
    r = ((mass_total + mass_moderator)/density_mix*3/4/np.pi)**(1/3)
    n = 1000  # Changed from 10000 to 1000
    initial_position = sphere_position(r, n)
    coeff = np.array([r])
    
    # Calculate atomic number densities
    N_natU = density_mix * N_A/238  # Using 238 as approximate molar mass for natural uranium
    N_graphite = (mass_moderator/(mass_total + mass_moderator)) * density_mix * N_A/12.01
    N = np.array([N_natU, 0, N_graphite])  # Middle value is 0 as we're not using U-238 separately
    
    # Fast neutron cross sections
    macro_sigma_total_fast = (df_U.loc["natU","σ_T"]*N_natU + 
                            df_moderator.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_fast = 1/macro_sigma_total_fast
    
    # Thermal neutron cross sections
    macro_sigma_total_thermal = (df_U_thermal.loc["natU","σ_T"]*N_natU + 
                               df_moderator_thermal.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_thermal = 1/macro_sigma_total_thermal
    
    # Resonance region cross sections
    macro_sigma_total_resonance = (df_U_resonance.loc["natU","σ_T"]*N_natU + 
                                 df_moderator_resonance.loc["Graphite","σ_T"]*N_graphite)*1e-24
    mf_path_resonance = 1/macro_sigma_total_resonance
    
    shape = 'sphere'
    return initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance

def plot_energy_distribution(df, generations_to_plot, reactor_type):
    """Plot energy distribution with immediate display"""
    plt.figure(figsize=(15, 6))
    plt.suptitle(f'Figure 1: Energy Distribution - {reactor_type}', y=1.02)
    
    # Create two subplots
    plt.subplot(1, 2, 1)
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # First subplot: Full energy range
    for i, gen in enumerate(generations_to_plot):
        gen_data = df[df['generation'] == gen]
        if len(gen_data) > 0:  # Only plot if there is data for this generation
            plt.hist(np.log10(gen_data['energy']), bins=50, alpha=0.5, 
                    label=f'Gen {gen}', color=colors[i % len(colors)])
            print(f"Generation {gen}: {len(gen_data)} neutrons")  # Debug information
    
    plt.axvline(x=np.log10(THERMAL_ENERGY), color='r', linestyle='--', 
                label=f'Thermal (<{THERMAL_ENERGY} eV)')
    plt.axvline(x=np.log10(RESONANCE_UPPER), color='g', linestyle='--', 
                label=f'Resonance (<{RESONANCE_UPPER} eV)')
    
    plt.xlabel('Log Energy (eV)')
    plt.ylabel('Count')
    plt.title('Full Energy Range')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Second subplot: Focus on thermal and resonance regions
    plt.subplot(1, 2, 2)
    for i, gen in enumerate(generations_to_plot):
        gen_data = df[df['generation'] == gen]
        thermal_data = gen_data[gen_data['energy'] <= RESONANCE_UPPER]
        if len(thermal_data) > 0:  # Only plot if there is thermal data for this generation
            plt.hist(np.log10(thermal_data['energy']), bins=30, alpha=0.5,
                    label=f'Gen {gen}', color=colors[i % len(colors)])
    
    plt.axvline(x=np.log10(THERMAL_ENERGY), color='r', linestyle='--',
                label=f'Thermal (<{THERMAL_ENERGY} eV)')
    plt.xlabel('Log Energy (eV)')
    plt.ylabel('Count')
    plt.title('Thermal and Resonance Region Detail')
    plt.xlim([-2, np.log10(RESONANCE_UPPER)])
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Update the display
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)

def plot_energy_regions(df, reactor_type):
    """Plot energy regions with immediate display"""
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Figure 2: Energy Region Distribution - {reactor_type}', y=1.02)
    
    # Calculate proportion of neutrons in each energy region by generation
    thermal_ratio = []
    resonance_ratio = []
    fast_ratio = []
    
    for gen in range(1, g+1):
        gen_data = df[df['generation'] == gen]
        if len(gen_data) > 0:
            thermal_count = len(gen_data[gen_data['energy_region'] == 'thermal'])
            resonance_count = len(gen_data[gen_data['energy_region'] == 'resonance'])
            fast_count = len(gen_data[gen_data['energy_region'] == 'fast'])
            total_count = len(gen_data)
            
            thermal_ratio.append(thermal_count / total_count * 100)
            resonance_ratio.append(resonance_count / total_count * 100)
            fast_ratio.append(fast_count / total_count * 100)
        else:
            thermal_ratio.append(0)
            resonance_ratio.append(0)
            fast_ratio.append(0)
    
    plt.plot(range(1, g+1), thermal_ratio, 'o-', color='blue', label='Thermal')
    plt.plot(range(1, g+1), resonance_ratio, 'o-', color='green', label='Resonance')
    plt.plot(range(1, g+1), fast_ratio, 'o-', color='red', label='Fast')
    plt.xlabel('Generation')
    plt.ylabel('Percentage of Neutrons (%)')
    plt.title('Proportion of Neutrons in Each Energy Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Update display immediately
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)

def plot_multiplication_factor(k_arr, dk_arr, reactor_type):
    """Plot multiplication factor with error bars in a line plot"""
    plt.figure(figsize=(10, 6))
    plt.suptitle(f'Figure 3: Multiplication Factor - {reactor_type}', y=1.02)
    
    # Plot the main line with error bars
    x = range(1, len(k_arr) + 1)
    plt.errorbar(x, k_arr, yerr=dk_arr, fmt='o-', color='blue', 
                capsize=5, capthick=1, ecolor='red', 
                markersize=8, markerfacecolor='blue')
    
    # Add critical line
    plt.axhline(y=1.0, color='r', linestyle='--', label='Critical (k=1)')
    
    # Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Multiplication Factor (k)')
    plt.title('Multiplication Factor Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Update display immediately
    plt.draw()
    plt.pause(0.001)
    plt.show(block=False)

def run_simulation(reactor_type, mass_total=60000, u235_fraction=0.016, moderator_fraction=600):
    """
    Run a complete reactor simulation
    
    Parameters:
    -----------
    reactor_type : str
        Type of reactor ("Pure U-235", "Mixed Uranium", or "Natural Uranium")
    mass_total : float
        Total uranium mass (g)
    u235_fraction : float
        Mass fraction of U-235 (between 0 and 1)
    moderator_fraction : float
        Mass ratio of moderator to total uranium (default: 600 for graphite)
    """
    logger.info(f"Starting simulation for {reactor_type}")
    logger.info(f"Parameters: mass={mass_total}g, U235 fraction={u235_fraction}, moderator ratio={moderator_fraction}")
    
    # Setup reactor
    if reactor_type == "Pure U-235":
        logger.info("Setting up Pure U-235 reactor")
        initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance = reactor_pure235_sphere(mass_total)
    elif reactor_type == "Mixed Uranium":
        logger.info("Setting up Mixed Uranium reactor")
        initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance = reactor_mixed_uranium_sphere(mass_total, u235_fraction, moderator_fraction)
    elif reactor_type == "Natural Uranium":
        logger.info("Setting up Natural Uranium reactor")
        initial_position, shape, coeff, mf_path_fast, mf_path_thermal, mf_path_resonance, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance = reactor_natural_uranium_sphere(mass_total, moderator_fraction)
    else:
        logger.error("Invalid reactor type")
        raise ValueError(f"Invalid reactor type: {reactor_type}")
    
    # Initialize neutrons
    df_neutron_walking = pd.DataFrame(columns=["x","y",'z','status','generation','energy','energy_region'], index=range(len(initial_position)))
    df_neutron_walking['generation'] = np.zeros(len(df_neutron_walking)) + 1
    df_neutron_walking['energy'] = np.zeros(len(df_neutron_walking)) + THERMAL_ENERGY
    df_neutron_walking['energy_region'] = 'thermal'
    df_neutron_walking.iloc[:,0:3] = initial_position
    
    df_neutron_total = pd.DataFrame()
    k_arr = np.array([])
    dk_arr = np.array([])
    
    # Early stopping threshold
    k_threshold = 0.01  # Stop if k falls below this value
    
    # Loop through generations
    for i in range(g):
        logger.info(f"Processing generation {i+1}")
        new_position, step_taken, df_neutron_Nowalking = process(
            df_neutron_walking, mf_path_fast, mf_path_thermal, mf_path_resonance,
            shape, coeff, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance
        )
        
        # Add debug information for tracking neutron counts
        print(f"Generation {i+1} - Neutrons before processing: {len(df_neutron_walking)}")
        print(f"Generation {i+1} - Neutrons after processing: {len(df_neutron_Nowalking)}")
        
        df_neutron_total = pd.concat([df_neutron_total, df_neutron_Nowalking], ignore_index=True)
        
        df_neutron_walking, k, dk = generate_children_neutron(
            df_neutron_Nowalking, N, macro_sigma_total_fast, macro_sigma_total_thermal, macro_sigma_total_resonance
        )
        
        # Track number of new neutrons generated
        print(f"Generation {i+1} - New neutrons generated: {len(df_neutron_walking)}")
        
        k_arr = np.append(k_arr, k)
        dk_arr = np.append(dk_arr, dk)
        
        logger.info(f"Generation {i+1} completed: k = {k:.3f} ± {dk:.3f}")
        
        # Early stopping condition
        if k < k_threshold:
            logger.warning(f"k value ({k:.3f}) below threshold ({k_threshold}). Stopping simulation.")
            break
    
    # Plot results
    generations_to_plot = list(range(1, len(k_arr) + 1))
    
    plot_energy_distribution(df_neutron_total, generations_to_plot, reactor_type)
    plot_energy_regions(df_neutron_total, reactor_type)
    plot_multiplication_factor(k_arr, dk_arr, reactor_type)
    
    logger.info(f"Simulation completed for {reactor_type}")
    return k_arr, dk_arr

# Main execution block
if __name__ == "__main__":
    print("Starting reactor simulation...")
    
    # Clear any existing plots
    plt.close('all')
    
    # Define global variable g (number of generations)
    g = 10  # 10 generations for simulation
    
    # Run simulations for all reactor types
    reactor_types = ["Pure U-235", "Mixed Uranium", "Natural Uranium"]
    results = {}
    
    try:
        for reactor_type in reactor_types:
            print(f"\nSimulating {reactor_type} reactor...")
            results[reactor_type] = run_simulation(reactor_type)
            print(f"Completed {reactor_type} simulation")
            plt.draw()  # Update the plot
        
        # Final comparison plot
        plt.figure(figsize=(10, 6))
        plt.suptitle('Final Multiplication Factors Comparison', y=1.02)
        
        final_k_values = [results[rt][0][-1] for rt in reactor_types]
        final_dk_values = [results[rt][1][-1] for rt in reactor_types]
        
        plt.bar(reactor_types, final_k_values, yerr=final_dk_values, capsize=5,
                ecolor='black', alpha=0.7)
        plt.axhline(y=1.0, color='r', linestyle='--', label='Critical (k=1)')
        plt.xlabel('Reactor Type')
        plt.ylabel('Final Multiplication Factor (k)')
        plt.title('Comparison of Final Multiplication Factors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        for i, (k, dk) in enumerate(zip(final_k_values, final_dk_values)):
            plt.text(i, k, f'k = {k:.3f}\nσ = {dk:.3f}',
                     ha='center', va='bottom')
        
        # Display the plot inline
        plt.draw()
        
    except Exception as e:
        print(f"An error occurred during simulation: {str(e)}")
        logger.error(f"Simulation failed: {str(e)}")
        plt.close('all')  # Clean up plots
        raise  # Re-raise the exception for debugging