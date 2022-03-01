"""
    Import all necessary Python libraries .
"""
import numpy as np

def resample(particle_Weights, num_Particles):
    """
    Resample particle poses and particle weights using Stratified and Systematic Resampling method.

    : Inputs (Data Type): 1.) particle_Weights (array)            - Weights of the particles
                          2.) num_Particles (int)                 - Number of particles

    : Output (Data Type): 1.) resampled_particle_Poses (array)    - Resampled particle poses
                          2.) resampled_particle_Weight (array)   - Resampled weights of the particles
    """
    # Initialize the resampled particle poses and weights
    resampled_particle_Poses = np.zeros((3, num_Particles))
    resampled_particle_Weights = np.ones((1, num_Particles)) * 1/num_Particles 

    # Extract the first weight
    particle_Weight = particle_Weights[0, 0]

    # Initialize counter
    j = 0

    # Run for loop to resample 
    for idx in range(0, num_Particles):

        # Uniformly distributed particle weights
        Beta = np.random.uniform(0, 1 / num_Particles) + idx/ num_Particles

        while Beta > particle_Weight:

            # Update count first
            j += 1
            # Update particle weight
            particle_Weight += particle_Weights[0, j]

        # Add ( µ^(j), 1/N) to the new set
        resampled_particle_Poses[:, idx] = resampled_particle_Poses[:, j]

    return resampled_particle_Poses, resampled_particle_Weights


        



