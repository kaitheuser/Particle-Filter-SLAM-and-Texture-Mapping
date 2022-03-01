"""
    Import all necessary Python libraries .
"""
import matplotlib
import matplotlib.pyplot as plt; plt.ion()
matplotlib.use('Qt5Agg')
import numpy as np
import pr2_utils

def display_Occupancy_Grid_Map(MAP, vehicle_trajectory, iteration):
    """
    Display Occupancy Grid Map.
    : Inputs (Data Type): 1.) MAP (dict)                    - Occupancy-Grid Map
                          2.) vehicle_trajectory (array)    - Vehicle Trajectory
                          3.) iteration                     - Iteration number
    : Output            : None
    """
    # Extract vehicle x-y trajectory
    vehicle_x_coords, vehicle_y_coords = vehicle_trajectory

    # Scale to Occupancy-grid map
    vehicle_x_coords = np.ceil((vehicle_x_coords - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    vehicle_y_coords = np.ceil((vehicle_y_coords - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # Filter out points that are out of the occupancy-grid map.
    ind_valid_vehicle_pts = np.logical_and(np.logical_and(np.logical_and((vehicle_x_coords > MAP['res']), (vehicle_y_coords > MAP['res'])),
                                          (vehicle_x_coords < MAP['sizex'])), (vehicle_y_coords < MAP['sizey']))

    # Determine whether it is a freespace or an obstacle by calculating the p.m.f.
    freespace = ((np.exp(MAP['map']) / (1 + np.exp(MAP['map']))) < 0.15).astype(np.int)
    obstacle = ((np.exp(MAP['map']) / (1 + np.exp(MAP['map']))) > 0.85).astype(np.int)

    # Update the Occupancy-grid Map
    freespace[vehicle_y_coords[ind_valid_vehicle_pts], vehicle_x_coords[ind_valid_vehicle_pts]] = 2
    obstacle[vehicle_y_coords[ind_valid_vehicle_pts], vehicle_x_coords[ind_valid_vehicle_pts]] = 1
    
    # Plot the Occupancy-grid Map
    plt.imshow(freespace, cmap = "gray")
    plt.title("Occupancy-grid Map")
    plt.xlabel("Width [m]")
    plt.ylabel("Length [m]")
    plt.show()
    plt.savefig('OGmapSLAM' + str(iteration) + '.png', bbox_inches='tight')
    plt.pause(0.001) # Prevent crashing



def transformation_matrix(RotMat, TransVec):
    """
    Convert to a 4 x 4 transformation matrix with the rotation matrix and translation vector

    : Inputs (Data Type): 1.) RotMat (array)    - 3 x 3 Rotation Matrix
                          2.) TransVec (array)  - 1 x 3 Translation Vector

    : Output (Data Type): 1.) TransfMat         - 4 x 4 Transformation Matrix 
    """
    # Reshape the translation vector
    TransVec = TransVec.reshape(1, len(TransVec))

    # Combine the rotation matrix and the translation vector.
    TransfMat = np.concatenate((RotMat, TransVec.T), axis = 1)
    TransfMat = np.vstack((TransfMat, np.array([0, 0, 0, 1])))

    return TransfMat

def filter_lidar_pts_cloud(lidar_angles, lidar_ranges, max_range_lidar, min_range_lidar):
    """
    Filters out invalid lidar observation points
    :Inputs (Data Type): 1.) lidar_angles (array)                       - Unfiltered LIDAR's observation angles [rad]
                         2.) lidat_ranges (array)                       - Unfiltered LIDAR's observation points
                         3.) max_range_lidar (float)                    - Maximum detectable distance range [m]
                         4.) min_range_lidar (float)                    - Minimum detectable distance range [m]

    :Output (Data Type): 1.) valid_lidar_angles (array)                 - Filtered LIDAR's observation angles [rad]
                         2.) valid_lidat_ranges (array)                 - Filtered LIDAR's observation points
    """
    # Filter out invalid indices
    indValid = np.logical_and((lidar_ranges < max_range_lidar),(lidar_ranges > min_range_lidar))
    valid_lidar_ranges = lidar_ranges[indValid]
    valid_lidar_angles = lidar_angles[indValid]

    return valid_lidar_angles, valid_lidar_ranges

def occupancy_grid_mapping(MAP, lidar_angles, lidar_ranges, RotMat_lidar2vehicle, TransVec_lidar2vehicle, particle_Pose_with_greatest_weight, num_Particles):
    """
    Update the occupancy-grid map log-odds.
    Steps: a.) Convert the scan to cartesian coordinates.
           b.) Transform the scan from the lidar frame to the body frame and then to the world frame.
           c.) Convert the scan to cells (via bresenham2D or cv2.drawContours) and update the map log-odds

    :Inputs (Data Type): 1.) MAP (dict)                                 - Initial occupancy-grid map
                         2.) lidar_angles (array)                       - LIDAR's observation angles [rad]
                         3.) lidar_ranges (array)                       - LIDAR's observation distance ranges [m]
                         4.) RotMat_lidar2vehicle (array)               - LIDAR's rotation matrix in the vehicle body frame
                         5.) TransVec_lidar2vehicle (array)             - LIDAR's translation vector in the vehicle body frame
                         6.) particle_Pose_with_greatest_weight (array) - Particle Pose with highest weight
                         7.) num_Particles                              - Number of Particles

    : Output (Data Type): MAP (dict)                                    - Updated occupancy-grid map
    """

    ## Convert the LIDAR scan to cartesian coordinates
    #-------------------------------------------------
    # Extract vehicle x-y coordinates and orientation
    x_pos_vehicle, y_pos_vehicle, orientation_vehicle = particle_Pose_with_greatest_weight

    # Determine the points that are at the end of the laser line
    lidar_end_pt_x = lidar_ranges * np.cos(lidar_angles)
    lidar_end_pt_y = lidar_ranges * np.sin(lidar_ranges)

    # Convert to 4D-Cartesian homogeneous coordinates.
    positions_vehicle = np.vstack((lidar_end_pt_x, 
                                   lidar_end_pt_y,
                                   np.full(len(lidar_angles), TransVec_lidar2vehicle[2]),
                                   np.ones(len(lidar_angles))))

    ## Transform the LIDAR scan from the lidar frame to the body frame and then to the world frame.
    #----------------------------------------------------------------------------------------------
    # Rotation matrix from vehicle body frame to world frame. (Rotation around z-axis)
    RotMat_vehicle2world = np.array([[np.cos(orientation_vehicle), -np.sin(orientation_vehicle), 0],
                                             [np.sin(orientation_vehicle), np.cos(orientation_vehicle), 0],
                                             [0, 0, 1]])
    TransVec_vehicle2world = np.array([x_pos_vehicle, y_pos_vehicle, 0])
    # Get the LIDAR to Vehicle Transformation Matrix
    TransfMat_lidar2vehicle = transformation_matrix(RotMat_lidar2vehicle, TransVec_lidar2vehicle)
    # Get the Vehicle to World Transformation Matrix
    TransfMat_vehicle2world = transformation_matrix(RotMat_vehicle2world, TransVec_vehicle2world)
    # Compute the vehicle positions with transformation matrix from LIDAR to World Frame.
    positions_vehicle = np.dot(TransfMat_vehicle2world, np.dot(TransfMat_lidar2vehicle, positions_vehicle))
    # Scale the LIDAR positiion in World Frame to the Occupancy-gird Map resolution
    lidar_start_pt_x = np.ceil((x_pos_vehicle - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    lidar_start_pt_y = np.ceil((y_pos_vehicle - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    # Scale all the LIDAR observation end points to Occupancy-gird Map resolution
    lidar_end_pt_x = np.ceil((positions_vehicle[0,:] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    lidar_end_pt_y = np.ceil((positions_vehicle[1,:] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    ## Convert the scan to cells (via bresenham2D or cv2.drawContours) and update the map log-odds
    #---------------------------------------------------------------------------------------------
    for idx in range(0, len(lidar_angles)):

        # Determine the cells that the lidar beams pass through by applying Bresenham's ray tracing algorithm in 2D from "pr2_utils.py".
        valid_lidar_pts = pr2_utils.bresenham2D(lidar_start_pt_x, lidar_start_pt_y, lidar_end_pt_x[idx], lidar_end_pt_y[idx])

        # Extract x-y valid LIDAR points.
        valid_lidar_x_coords, valid_lidar_y_coords = valid_lidar_pts.astype(np.int16)
        
        # Filter out LIDAR points that are out of the occupancy-grid map.
        ind_valid_lidar_pts = np.logical_and( np.logical_and(np.logical_and((valid_lidar_x_coords > MAP['res']), (valid_lidar_y_coords > MAP['res'])),
                                            (valid_lidar_x_coords < MAP['sizex'])), (valid_lidar_y_coords < MAP['sizey']))

        # Update the map log-odds
        MAP['map'][valid_lidar_y_coords[ind_valid_lidar_pts], valid_lidar_x_coords[ind_valid_lidar_pts]] += np.log(1/4)

        # Increase the log-odds if the cell was observed occupied
        if ((lidar_end_pt_x[idx] > MAP['res']) and (lidar_end_pt_y[idx] > MAP['res']) and (lidar_end_pt_x[idx] < MAP['sizex']) and (lidar_end_pt_y[idx] < MAP['sizey'])):
            MAP['map'][lidar_end_pt_y[idx], lidar_end_pt_x[idx]] += 2 * np.log(4)

    # Constrain λMIN ≤ λi,t ≤ λMAX to avoid overconfident estimation
    MAP['map'] = np.clip(MAP['map'], num_Particles * np.log(1/4), num_Particles * np.log(4))

    return MAP






    



