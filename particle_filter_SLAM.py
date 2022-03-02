"""
    Import all necessary Python libraries .
"""
import cv2 as cv
import math
import matplotlib.pyplot as plt
import mapping
import numpy as np
import os
import prediction_update
import pr2_utils
import resampling
from tqdm import tqdm
import texture_mapping

"""
    Debugger Settings
"""
debug_array_Shape = False
debug_list_Length = False
debug_initialized_map = False
debug_sync_timestamps = False

"""
    Texture Mapping Setting (True == On) (False == Off)
"""
texture_mapping_settings = True                      

"""
    Load and initialize data and parameters.
"""
### Data-preprocessing ###
#-------------------------
## Calibrated Encoders Data
encoder_time, encoder_data = pr2_utils.read_data_from_csv('data/sensor_data/encoder.csv')                   # Time [nanoseconds]
left_encoder_counts = encoder_data[:, 0]
right_encoder_counts = encoder_data[:, 1]
num_Encoder_data, _ = encoder_data.shape 
if debug_array_Shape:
    print("Encoder Data Shape: " + str(encoder_data.shape))                                                 # Encoder Data Shape: (116048, 2)

## Light Detection and Ranging (LIDAR) Data
lidar_time, lidar_data = pr2_utils.read_data_from_csv('data/sensor_data/lidar.csv')
num_LIDAR_data, num_LIDAR_scanned_pts_per_sweep = lidar_data.shape 
if debug_array_Shape:
    print("LIDAR Data Shape: " + str(lidar_data.shape))                                                     # LIDAR Data Shape: (115865, 286)

## Stereo Camera Timestamps Data 
cam_time_left = os.listdir('data/stereo_images/stereo_left')                                                # Load the left image names
cam_time_right = os.listdir('data/stereo_images/stereo_right')                                              # Load the right image names
assert len(cam_time_left) == len(cam_time_right)
num_Cam_data = len(cam_time_left)                                                                           # Number of images
for idx in range(0, len(cam_time_left)):
    cam_time_left[idx] = str(cam_time_left[idx]).replace('.png','')                                         # Delete image file format
    cam_time_right[idx] = str(cam_time_right[idx]).replace('.png','')                                       # Delete image file format
cam_time_left = list(map(int, cam_time_left))                                                               # Convert str to int
cam_time_right = list(map(int, cam_time_right))                                                             # Convert str to int
cam_time_left = np.array(cam_time_left)                                                                     # Convert to array from list
cam_time_right = np.array(cam_time_right)                                                                   # Convert to array from list
F_f = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])                                                         # Image Flip, F_f

# Left Camera (yaml file parameters)
left_cam = cv.FileStorage('data/param/left_camera.yaml', cv.FILE_STORAGE_READ)                              # Load yaml file
left_cam_img_height = left_cam.getNode("image_height").real()                                               # Image Height
left_cam_img_width = left_cam.getNode("image_width").real()                                                 # Image Width
left_cam_mat = left_cam.getNode("camera_matrix").mat()                                                      # Pixel Scaling Matrix, K_s
left_cam_mat[0:2,2] = np.rint(left_cam_mat[0:2,2])                                                          # Round c_u and c_v to nearest int
left_cam_distortion_coeffs = left_cam.getNode("distortion_coefficients").mat()                              # Distortion Coefficients
left_cam_rectification_mat = left_cam.getNode("rectification_matrix").mat()                                 # Rectification Matrix (Not Used)
left_cam_projection_mat = left_cam.getNode("projection_matrix").mat()                                       # Intrinsic Matrix [K|0]
K_intrinsic_left = left_cam_projection_mat[:,:3]                                                            # K matrix [fsu, fstheta, c_u, 0;
                                                                                                            #           0,    fsv   , c_v, 0;
                                                                                                            #           0,     0,     0, fs_ub]

# Right Camera (yaml file parameters)
right_cam = cv.FileStorage('data/param/right_camera.yaml', cv.FILE_STORAGE_READ)                            # Load yaml file
right_cam_img_height = right_cam.getNode("image_height").real()                                             # Image Height
right_cam_img_width = right_cam.getNode("image_width").real()                                               # Image Width
right_cam_mat = right_cam.getNode("camera_matrix").mat()                                                    # Pixel Scaling Matrix, K_s
right_cam_mat[0:2,2] = np.rint(right_cam_mat[0:2,2])                                                        # Round c_u and c_v to nearest int
right_cam_distortion_coeffs = right_cam.getNode("distortion_coefficients").mat()                            # Distortion Coefficients
right_cam_rectification_mat = right_cam.getNode("rectification_matrix").mat()                               # Rectification Matrix (Not Used)
right_cam_projection_mat = right_cam.getNode("projection_matrix").mat()                                     # Intrinsic Matrix [K|0]
K_intrinsic_right = right_cam_projection_mat[:,:3]                                                          # K matrix

if debug_list_Length:
    print("Number of Stereo Camera Images: " + str(num_Cam_data))                                          # Number of Stereo Camera Images: 1161

## Fiber Optic Gyro (FOG) Data
fog_time, fog_data = pr2_utils.read_data_from_csv('data/sensor_data/fog.csv')
fog_roll_motion = fog_data[:, 0]
fog_pitch_motion = fog_data[:, 1]
fog_yaw_motion = fog_data[:, 2]
num_FOG_data, _ = fog_data.shape 
if debug_array_Shape:
    print("FOG Data Shape: " + str(fog_data.shape))                                                         # FOG Data Shape: (1160508, 3)

### Parameters Initialization ###
#--------------------------------
## Calibrated Encoders' Parameters from "EncoderParameter.txt" file
encoder_res = 4096                                                      # Encoder Resolution [ticks per second] or [Hz]
encoder_left_wheel_diameter =  0.623479                                 # Encoder Left Wheel Diameter [m]
encoder_right_wheel_diameter = 0.622806                                 # Encoder Right Wheel Diameter [m]
encoder_wheel_base = 1.52439                                            # Encoder Wheel Base (Distance Between Front and Back Wheels) [m]

## Light Detection and Ranging (LIDAR) Extrinsic Parameters from "Vehicle2Lidar.txt" file
RotMat_lidar2vehicle = np.array([[0.00130201, 0.796097, 0.605167],      # Vehicle to LIDAR Rotation Matrix in Vehicle Body Frame
                                 [0.999999, -0.000419027, -0.00160026],
                                 [-0.00102038, 0.605169, -0.796097]])
TransVec_lidar2vehicle = np.array([0.8349, -0.0126869, 1.76416])        # Vehicle to LIDAR Translation Vector in Vehicle Body Frame
# Other LIDAR parameters
FOV_lidar = 190                                                       # LIDAR field of view [Degrees]
start_angle_lidar = -5                                                # LIDAR FOV starting angle [Degrees]
end_angle_lidar = 185                                                 # LIDAR FOV ending angle [Degrees]
angular_res_lidar = 0.666                                             # LIDAR FOV angular resolution [Degrees]
max_range_lidar = 80                                                  # LIDAR maximum detecting range [m]
min_range_lidar = 2                                                   # LIDAR minimum detecting range [m]

## Stereo Camera Extrinsic and Intrinsic Parameters from "stereo_param.txt" and "Vehicle2Stereo.txt" file
# Extrinsic Parameters
RotMat_cam2vehicle = np.array([[-0.00680499, -0.0153215, 0.99985],      # Vehicle to Stereo Camera Rotation Matrix in Vehicle Body Frame (oRr)
                               [-0.999977, 0.000334627, -0.00680066],
                               [-0.000230383, -0.999883, -0.0153234]])
TransVec_cam2vehicle = np.array([1.64239, 0.247401, 1.58411])           # Vehicle to Stereo Camera Translation Vector in Vehicle Body Frame
# Intrinsic Parameter
baseline_cam = 475.143600050775/1000                                    # Baseline distance (Distance between two cameras) [m]

## Fiber Optic Gyro (FOG) Extrinsic Parameters from "Vehicle2FOG.txt" file
RotMat_FOG2vehicle = np.eye(3)                                          # Vehicle to FOG Rotation Matrix in Vehicle Body Frame
TransVec_FOG2vehicle = np.array([-0.335, -0.035, 0.78])                 # Vehicle to FOG Translation Vector in Vehicle Body Frame


### Particle Set and Weights Initialization ###
#----------------------------------------------
num_Particles = 100                                                     # Number of Particles in a Particle Set
particle_Poses_arr = np.zeros((3, num_Particles))                       # Initialize an array that stores particle positions and orientations [x [m], y [m], orientation [rad]]^T
particle_Weights_arr = np.ones((1, num_Particles)) * 1/num_Particles    # Initialize a vector that stores particle weights
particle_Pose_with_greatest_weight = np.zeros(3)                        # Initialize the particle pose with greatest weight as [0, 0, 0]
num_Eff_particles_threshold = num_Particles * 0.2                       # Number of Effective Particles Threshold (Resampling)


### Occupancy-Grid Map Initialization ###
#----------------------------------------
## Initialize the map (follow the map initialization format from "pr2_utils.py" test_mapCorrelation() function)
MAP = {}
MAP['res']   = 1                                                                                    # Map Resolution / Step [m]
MAP['xmin']  = -150                                                                                 # Map x-min [m]
MAP['ymin']  = -1350                                                                                # Map y-min [m]
MAP['xmax']  =  1450                                                                                # Map x-max [m]
MAP['ymax']  =  150                                                                                 # Map y-max [m]
MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))                          # Number of Cells in a Row
MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))                          # Number of Cells in a Column
MAP['map'] = np.zeros((MAP['sizey'], MAP['sizex']),dtype=np.float32)                                # DATA TYPE: float32 (if not can't work with np.log(4))

## Get the first laser scan to initialize the map
lidar_angles = np.linspace(start_angle_lidar, end_angle_lidar, math.ceil(abs((end_angle_lidar - start_angle_lidar)/angular_res_lidar))) / 180 * np.pi
first_sweep_lidar_ranges = lidar_data[0, :]

# Filter out invalid indices
valid_lidar_angles, valid_first_sweep_lidar_ranges = mapping.filter_lidar_pts_cloud(lidar_angles, first_sweep_lidar_ranges, max_range_lidar, min_range_lidar)

# Use the first laser scan to initialize the map
MAP = mapping.occupancy_grid_mapping(MAP, valid_lidar_angles, valid_first_sweep_lidar_ranges, RotMat_lidar2vehicle, TransVec_lidar2vehicle, particle_Pose_with_greatest_weight, num_Particles)

# Check whether the initialized map makes sense.
if debug_initialized_map:
    # Plot the Intialized Occupancy Grid Map
    first_map = plt.figure()
    plt.imshow(MAP['map'], cmap="gray")
    plt.title("Occupancy grid map")
    plt.xlabel("Width [m]")
    plt.ylabel("Length [m]")
    plt.show(block=True)


### Prediction, Update, and Resample (PUR) Iterations Process ###
#----------------------------------------------------------------
# Initialize synced encoder data and synced LIDAR data with FOG data.
synced_left_encoder_counts = np.zeros(num_FOG_data)
synced_lidar_data = np.zeros((num_FOG_data, num_LIDAR_scanned_pts_per_sweep))
# Initialize trajectory array that contains all the trajectory indices of the vehicle
vehicle_trajectory = np.zeros((2, 1))
# Initialize RGB Map for Texture Mapping
RGB_map = np.ones((MAP['sizey'], MAP['sizex'],3))
for height in range(0, MAP['sizey']):
    for width in range(0, MAP['sizex']):
        RGB_map[height, width, :] = np.array([192/255,192/255,192/255])

# Start interating through the FOG data timestamps
for idx in tqdm(range(0, num_FOG_data)):

    ## Sync the timestamps of the encoder and LIDAR with the FOG timestamps
    # Find which index has the minimum difference
    idx_min_encoder = np.argmin(abs(encoder_time - fog_time[idx]))
    idx_min_lidar = np.argmin(abs(lidar_time - fog_time[idx]))
    idx_min_cam_left = np.argmin(abs(cam_time_left - fog_time[idx]))
    idx_min_cam_right = np.argmin(abs(cam_time_right - fog_time[idx]))
    # Store the synced data
    synced_left_encoder_counts[idx] = left_encoder_counts[idx_min_encoder]
    synced_lidar_data[idx, :] = lidar_data[idx_min_lidar, :]
    synced_cam_time_left = cam_time_left[idx_min_cam_left]
    synced_cam_time_right = cam_time_right[idx_min_cam_right]

    # Debug synced data
    if debug_sync_timestamps and idx == 0:
        # Print the timestamps of all sensors
        print("FOG Time [ns]: " +str(fog_time[idx]) + ", Encoder Time [ns]: " + str(encoder_time[idx_min_encoder]) + ", LIDAR Time [ns]: " + str(lidar_time[idx_min_lidar]))

    ## Predict the vehicle trajectory by calculating the angular and linear velocities with the FOG data and Encoder data, respectively.
    # Calculate the change in rotation angle
    delta_Rot = fog_yaw_motion[idx]
    # Calculate the change in displacement
    if idx == 0:
        delta_Dist = math.pi * encoder_left_wheel_diameter * synced_left_encoder_counts[idx] / encoder_res
    else:
        delta_Dist = math.pi * encoder_left_wheel_diameter * (synced_left_encoder_counts[idx] - synced_left_encoder_counts[idx - 1]) / encoder_res
    # Predict the particle poses.
    particle_Poses_arr = prediction_update.predict(particle_Poses_arr, delta_Rot, delta_Dist, num_Particles)

    ## Update only every 10 iterations because LIDAR measurement frequency is 10 times lesser than FOG measurement frequency.
    if idx % 100 == 0 or idx == num_FOG_data-1:

        ## Get the laser scan to initialize the map
        lidar_ranges = synced_lidar_data[idx, :]

        # Filter out invalid LIDAR observation points.
        valid_lidar_angles, valid_lidar_ranges = mapping.filter_lidar_pts_cloud(lidar_angles, lidar_ranges, max_range_lidar, min_range_lidar)

        # Update the weights of the particle
        particle_Weights_arr = prediction_update.update(particle_Poses_arr, particle_Weights_arr, num_Particles, MAP, valid_lidar_angles, valid_lidar_ranges, RotMat_lidar2vehicle, TransVec_lidar2vehicle)

        # Select the particle with the greatest weight
        particle_Pose_with_greatest_weight = particle_Poses_arr[:, np.argmax(particle_Weights_arr)]

        # Update the trajectory of the vehicle.
        vehicle_trajectory = np.hstack((vehicle_trajectory, particle_Pose_with_greatest_weight[0:2].reshape(2,1)))

        # Update the occupancy-grid map
        MAP = mapping.occupancy_grid_mapping(MAP, valid_lidar_angles, valid_lidar_ranges, RotMat_lidar2vehicle, TransVec_lidar2vehicle, particle_Pose_with_greatest_weight, num_Particles)

        ## Texture Mapping
        if texture_mapping_settings and idx % 100 == 0 or idx == num_FOG_data-1:
            RGB_map = texture_mapping.texture_map(RGB_map, MAP, RotMat_cam2vehicle, TransVec_cam2vehicle, particle_Pose_with_greatest_weight, K_intrinsic_left, K_intrinsic_right, synced_cam_time_left, synced_cam_time_right, baseline_cam, idx, num_FOG_data, left_cam_distortion_coeffs, right_cam_distortion_coeffs)

        ## Resampling process if required
        # Calculate the number of effective particles.
        num_particles_eff = 1 / np.dot(particle_Weights_arr, particle_Weights_arr.reshape(num_Particles, 1))
        # If the number of effective particles is low, resample.
        if num_particles_eff <= num_Eff_particles_threshold:
            particle_Poses_arr, particle_Weights_arr = resampling.resample(particle_Weights_arr, num_Particles)

        ## Display Occupancy-grid map
        if idx == 10000 or idx == 100000 or idx == 300000 or idx == 500000 or idx == 700000 or idx == 1000000 or idx == num_FOG_data-1:
            mapping.display_Occupancy_Grid_Map(MAP, vehicle_trajectory, idx)