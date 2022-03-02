"""
    Import all necessary Python libraries .
"""
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt; plt.ion()
matplotlib.use('Qt5Agg')
import numpy as np


def texture_map(RGB_map, MAP, RotMat_cam2vehicle, TransVec_cam2vehicle, particle_Pose_with_greatest_weight, K_intrinsic_left, K_intrinsic_right, synced_cam_time_left, synced_cam_time_right, baseline, iteration, num_FOG_data, distortion_coeff_left, distortion_coeff_right):
    """
    Compute a disparity image from stereo image pairs using the provided script in p2 utils.py and estimate depth for each pixel via the stereo camera model. 
    Project colored points from the left camera onto your occupancy grid in order to color it. 

    : Inputs (Data Type): 1.) RGB_map (array)                            - Occupancy-grid map with colors
                          2.) MAP (dict)                                 - Initial occupancy-grid map
                          3.) RotMat_cam2vehicle (array)                 - Stereo Camera's rotation matrix in the vehicle body frame
                          4.) TransVec_cam2vehicle (array)               - Stereo Camera's translation vector in the vehicle body frame
                          5.) particle_Pose_with_greatest_weight (array) - Particle Pose with highest weight
                          6.) K_intrinsic_left (array)                   - Intrinsic Parameters Matrix, K for the left stereo camera
                          7.) K_intrinsic_right (array)                  - Intrinsic Parameters Matrix, K for the right stereo camera
                          8.) synced_cam_time_left (int)                 - Index that has the synced camera timestamp (left)
                          9.) synced_cam_time_right (int)                - Index that has the synced camera timestamp (right)
                          10.) baseline (float)                          - Baseline in m
                          11.) iteration (int)                           - Number of iterations
                          12.) num_FOG_data (int)                        - Number of FOG data
                          13.) distortion_coeff_left                     - Left Stereo Camera Distortion coefficients
                          14.) distortion_coeff_right                    - Right Stereo Camera Distortion coefficients

    : Output (Data Type): 1.) RGB_map (array)                            - Occupancy-grid map with colors
    """
    # Extract vehicle x-y coordinates and orientation
    x_pos_vehicle, y_pos_vehicle, orientation_vehicle = particle_Pose_with_greatest_weight

    # Rotation matrix from vehicle body frame to world frame. (Rotation around z-axis)
    RotMat_vehicle2world = np.array([[np.cos(orientation_vehicle), -np.sin(orientation_vehicle), 0],
                                     [np.sin(orientation_vehicle), np.cos(orientation_vehicle), 0],
                                     [0, 0, 1]])
    TransVec_vehicle2world = np.array([x_pos_vehicle, y_pos_vehicle, 0])

    ## Compute Disparity
    # Left Stereo Image Path
    path_l = 'data/stereo_images/stereo_left/' + str(synced_cam_time_left) + '.png'
    # Right Stereo Image Path
    path_r = 'data/stereo_images/stereo_right/' + str(synced_cam_time_right) + '.png'

    # Read Left Stereo Image
    image_l = cv.imread(path_l, 0)
    # Read Right Stereo Image
    image_r = cv.imread(path_r, 0)

    # Skip if the timestamps are not the same
    if image_l is None or image_r is None:
        # Plot the Occupancy-grid Map
        plt.imshow(RGB_map)
        plt.title("RGB Occupancy-grid Map")
        plt.xlabel("Width [m]")
        plt.ylabel("Length [m]")
        plt.show()
        plt.savefig('OGmapRGBfree' + str(iteration) + '.png', bbox_inches='tight')
        plt.pause(0.001) # Prevent crashing
        return RGB_map

    # Undistort images
    image_l = cv.undistort(image_l, K_intrinsic_left, distortion_coeff_left, None)
    image_r = cv.undistort(image_r, K_intrinsic_right, distortion_coeff_right, None)

    # Convert to RGB
    image_l_rgb = cv.cvtColor(image_l, cv.COLOR_BAYER_BG2RGB)
    image_r_rgb = cv.cvtColor(image_r, cv.COLOR_BAYER_BG2RGB)
    # Convert to Grayscale
    image_l_gray = cv.cvtColor(image_l_rgb, cv.COLOR_RGB2GRAY)
    image_r_gray = cv.cvtColor(image_r_rgb, cv.COLOR_RGB2GRAY)

    # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=9) 
    disparity = stereo.compute(image_l_gray, image_r_gray)

    ## Estimate Depth, z
    # Extract the fs_u
    fs_u = K_intrinsic_left[0, 0]
    fs_theta = K_intrinsic_left[0, 1]
    c_u = K_intrinsic_left[0, 2]
    fs_v = K_intrinsic_left[1, 1]
    c_v = K_intrinsic_left[1, 2]
    # Calculate depth z in every pixel
    with np.errstate(divide='ignore'):
        depth_flat = 1 / disparity.flatten() * fs_u * baseline

    # Initialize Pixel Coordinate
    u_L = np.tile(np.arange(0, image_l_rgb.shape[1]), (image_l_rgb.shape[0], 1)).flatten()
    v_L = np.tile(np.array([np.arange(0, image_l_rgb.shape[0])]).transpose(), (1, image_l_rgb.shape[1])).flatten()
    pixel_coords = np.vstack((u_L, v_L))

    # Optical frame y-coordinates
    y_coords_optical = depth_flat * (v_L - c_v) / fs_v

    # Optical frame x-coordinates
    with np.errstate(invalid='ignore'):
        x_coords_optical = (depth_flat * (u_L - c_u) - fs_theta * y_coords_optical) / fs_u

    # Optical frame coordinates
    optical_frame_coords = np.vstack((x_coords_optical, y_coords_optical, depth_flat))

    # Pixel in world coordinates
    world_coords = np.dot(np.linalg.inv(np.dot(RotMat_cam2vehicle.T, RotMat_vehicle2world.T)), optical_frame_coords) + TransVec_vehicle2world.reshape(-1,1)

    # Valid indices based on the camera height threshold
    valid_ind = np.where(world_coords[2,:] <=  TransVec_cam2vehicle[2])
    valid_pixel_coords = pixel_coords[:, valid_ind[0]]
    valid_world_coords = world_coords[:, valid_ind[0]]
    valid_pixel_coords_RGB_values = np.zeros((valid_pixel_coords.shape[1],3))
    for idx in range(0, valid_pixel_coords.shape[1]):
        valid_pixel_coords_RGB_values[idx, :] = image_l_rgb[valid_pixel_coords[1, idx], valid_pixel_coords[0, idx],:]

    # Extract vehicle x-y world coordinates
    x_coords_world, y_coords_world, _ = valid_world_coords

    # Scale to Occupancy-grid map
    x_coords_world = np.ceil((x_coords_world - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    y_coords_world = np.ceil((y_coords_world - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

    # Determine whether it is a freespace or an obstacle by calculating the p.m.f.
    freespace = ((np.exp(MAP['map']) / (1 + np.exp(MAP['map']))) < 0.15).astype(np.int)

    # Update the RGB Occupancy-grid Map
    for idx in range(0, valid_pixel_coords_RGB_values.shape[0]):
        if freespace[y_coords_world[idx], x_coords_world[idx]] != 0:
            RGB_map[y_coords_world[idx], x_coords_world[idx], :] = valid_pixel_coords_RGB_values[idx,:].astype(np.int16)/int(255)

    ## Display Texture map
    if iteration == 10000 or iteration == 100000 or iteration == 300000 or iteration == 500000 or iteration == 700000 or iteration == 1000000 or iteration == num_FOG_data-1:

        # Plot the Occupancy-grid Map
        plt.imshow(RGB_map)
        plt.title("RGB Occupancy-grid Map")
        plt.xlabel("Width [m]")
        plt.ylabel("Length [m]")
        plt.show()
        plt.savefig('OGmapRGB' + str(iteration) + '.png', bbox_inches='tight')
        plt.pause(0.001) # Prevent crashing

    return RGB_map

    









