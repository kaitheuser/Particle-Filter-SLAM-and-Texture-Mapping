"""
    Import all necessary Python libraries .
"""

import numpy as np
import pr2_utils
import mapping

def predict(particle_Poses, delta_Rot, delta_Dist, num_Particles):
    """
    Predict the next step of position x at t+1

    : Inputs (Data Type): 1.) particle_Poses (array)              - Particles with coordinates and orientations in world frame 3 x N matrix
                          2.) delta_Rot (float)                   - Change in rotation
                          3.) delta_Dist (float)                  - Change in displacement

    : Output (Data Type): 1.) predicted_particle_Poses (array)    - Predicted Particles with coordinates and orientations 3 x N matrix at t+1
    """

    # Extract current vehicle pose at t
    x_pos_vehicle, y_pos_vehicle, orientation_vehicle = particle_Poses
    # Get the number of particles
    #_, num_particles = particle_Poses.shape

    ## Use the differential-drive model with velocity from the encoders and angular velocity from the gyroscope to predict the motion of each particle and add noise.
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Updated rotation angle
    theta = orientation_vehicle + delta_Rot
    # Change in displacement in x-direction.
    delta_dist_x = delta_Dist * np.cos(theta)
    # Change in displacement in y-direction.
    delta_dist_y = delta_Dist * np.sin(theta)
    # Predict the each particle position and orientation for t+1 with Gaussian noise
    x_pos_vehicle += (delta_dist_x + (np.array([np.random.normal(0, abs(np.max(delta_dist_x))/num_Particles)]))[0])
    y_pos_vehicle += (delta_dist_y + (np.array([np.random.normal(0, abs(np.max(delta_dist_y))/num_Particles)]))[0])
    orientation_vehicle += (delta_Rot + (np.array([np.random.normal(0, abs(np.max(delta_Rot))/num_Particles)]))[0])
    # Predicted vehicle pose in world frame at t+1
    predicted_particle_Poses = np.vstack((x_pos_vehicle, y_pos_vehicle, orientation_vehicle))

    return predicted_particle_Poses

def update(particle_Poses, particle_Weights, num_Particles, MAP, lidar_angles, lidar_ranges, RotMat_lidar2vehicle, TransVec_lidar2vehicle):
    """
    Combines Robot State and Map Update.
    a.) Use the laser scan from each particle to compute map correlation (via getMapCorrelation) and update the particle weights.
    b.) Choose the particle with largest weight α(k)t|t, project the laser scan zt to the world frame and update the map log-odds.

    : Inputs (Data Type): 1.) particle_Poses (array)              - Particles with coordinates and orientations in world frame 3 x N matrix
                          2.) particle_Weights (array)            - Weights of the particles
                          3.) num_Particles (int)                 - Number of particles
                          3.) MAP (dict)                          - Initial occupancy-grid map
                          4.) lidar_angles (array)                - LIDAR's observation angles [rad]
                          5.) lidar_ranges (array)                - LIDAR's observation distance ranges [m]
                          6.) RotMat_lidar2vehicle (array)        - LIDAR's rotation matrix in the vehicle body frame
                          7.) TransVec_lidar2vehicle (array)      - LIDAR's translation vector in the vehicle body frame

    : Output (Data Type): 1.) updated_particle_Weights (array)    - Updated weights of the particles
    """

    ## Use the laser scan from each particle to compute map correlation (via getMapCorrelation) and update the particle weights.
    # Initialize all the cells x-y positions in the map.
    x_ind_map = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])
    y_ind_map = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res'])

    # Initialize x_range and y_range for the getMapCorrelation.
    x_range = np.arange(-4 * MAP['res'] , 4 * MAP['res'] + MAP['res'], MAP['res'])
    y_range = np.arange(-4 * MAP['res'] , 4 * MAP['res'] + MAP['res'], MAP['res'])

    # Initialize vector that store correlation of each particle
    particle_correlations = np.zeros(num_Particles)

    # Determine the p.m.f of the map grid cells with logistic sigmoid function
    grid_cell_pmfs = ((np.exp(MAP['map'])/(1 + np.exp(MAP['map']))) < 0.5).astype(np.int)

    # Determine the points that are at the end of the laser line
    lidar_end_pt_x = lidar_ranges * np.cos(lidar_angles)
    lidar_end_pt_y = lidar_ranges * np.sin(lidar_angles)

    # LIDAR homogeneous positions in vehicle frame
    lidar_homo_coords = np.vstack((lidar_end_pt_x, 
                                   lidar_end_pt_y,
                                   np.full(len(lidar_angles), TransVec_lidar2vehicle[2]),
                                   np.ones(len(lidar_angles))))

    # Transformation matrix from lidar to vehicle
    TransfMat_lidar2vehicle = mapping.transformation_matrix(RotMat_lidar2vehicle, TransVec_lidar2vehicle)

    ## Choose the particle with largest weight α(k)t|t, project the laser scan zt to the world frame and update the map log-odds.
    # Update the particles pose iteratively
    for idx in range(0, particle_Poses.shape[1]):

        # Extract a particle pose.
        x_pos_particle, y_pos_particle, orientation_particle = particle_Poses[:, idx]

        # Rotation matrix from vehicle body frame to world frame. (Rotation around z-axis)
        RotMat_vehicle2world = np.array([[np.cos(orientation_particle), -np.sin(orientation_particle), 0],
                                         [np.sin(orientation_particle), np.cos(orientation_particle), 0],
                                         [0, 0, 1]])
        TransVec_vehicle2world = np.array([x_pos_particle, y_pos_particle, 0])
        # Transformation matrix from vehicle to world
        TransfMat_vehicle2world = mapping.transformation_matrix(RotMat_vehicle2world, TransVec_vehicle2world)
        # Compute the particle positions with transformation matrix from LIDAR to World Frame.
        particle_positions = np.dot(TransfMat_vehicle2world, np.dot(TransfMat_lidar2vehicle, lidar_homo_coords))

        # Occupied x,y positions from range sensor (in physical unit) 
        vp = np.vstack((particle_positions[0, :], particle_positions[1, :]))

        # Determine the map correlation
        MAP_corr = pr2_utils.mapCorrelation(grid_cell_pmfs, x_ind_map, y_ind_map, vp, x_range, y_range)

        # Determine the index with the largest correlation value.
        particle_correlations[idx] = np.max(MAP_corr)

    # Update particle weights using softmax
    softmax = np.exp(particle_correlations - np.max(particle_correlations)) / np.sum(np.exp(particle_correlations - np.max(particle_correlations)))
    updated_particle_Weights = particle_Weights * softmax / np.sum(particle_Weights * softmax)

    return updated_particle_Weights