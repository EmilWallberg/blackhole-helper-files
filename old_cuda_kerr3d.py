import ctypes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

from math import cos, sin, sqrt, atan2, acos

# Constants
speed_of_light = 299792458
M = 1
mass = 8.543e36

a = 0.99
Newton_G = 6.67408e-11
r_s = 2 * Newton_G * mass / speed_of_light**2
A = a * r_s / 2  # Might be unit conversion from SI units to Natural units
epsilon = 1e-10

r_env = r_s * 15

# --- Parameters and settings ---
# Define the grid dimensions for the angular variation.
n_theta = 100     # Number of different polar angles (θ)
n_phi = 50      # Number of different azimuthal angles (φ)
n_rays = n_theta * n_phi   # Total ray count
max_steps = 10000
state_dim = 5   # [r, theta, phi, p_r, p_theta]

# Spherical to Cartesian conversion (r, θ, φ -> x, y, z)
def spherical_to_cartesian(radius, polar_angle, azimuthal_angle):
    x = radius * np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = radius * np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = radius * np.cos(polar_angle)
    return np.array([x, y, z])

# Boyer-Lindquist coordinates to Cartesian (R, θ, φ -> x, y, z)
def boyer_lindquist_to_cartesian(R, T, P):
    x = np.sqrt(R**2 + A**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2 + A**2) * np.sin(T) * np.sin(P)
    z = R * np.cos(T)
    return np.array([x, y, z])

# Inverse conversion: Cartesian (x, y, z) -> Boyer-Lindquist (R, θ, φ)
def cartesian_to_boyer_lindquist(x, x_velocity, y, y_velocity, z, z_velocity):
    azimuthal_angle = np.arctan2(y, x)  # Azimuthal angle (φ)
    radius = np.sqrt((-A**2 + x**2 + y**2 + z**2 +
                     np.sqrt(A**2 * (A**2 - 2*x**2 - 2*y**2 + 2*z**2) +
                             (x**2 + y**2 + z**2)**2)) / 2)
    polar_angle = np.arccos(z / radius)  # Polar angle (θ)
    
    # Radial velocity and derivatives
    radius_velocity = (radius * (x * x_velocity + y * y_velocity + z * z_velocity)) / \
                      (2 * radius**2 + A**2 - x**2 - y**2 - z**2) + \
                      A**2 * z * z_velocity / (radius * (2 * radius**2 + A**2 - x**2 - y**2 - z**2))
    polar_angle_velocity = (z * radius_velocity - z_velocity * radius) / (radius * sqrt(radius**2 - z**2))
    azimuthal_angle_velocity = (y_velocity * x - x_velocity * y) / (x**2 + y**2)
    
    return np.array([radius, radius_velocity, polar_angle, polar_angle_velocity, azimuthal_angle, azimuthal_angle_velocity])

def rotate_x(x, y, z, v=np.pi/2):
    # 90 degrees around the x-axis
    y_rot = y * cos(v) - z * sin(v)
    z_rot = y * sin(v) + z * cos(v)
    return x, y_rot, z_rot

def rotate_z(x, y, z, v=np.pi/2):
    # -90 degrees around the z-axis
    x_rot = x * cos(v) - y * sin(v)
    y_rot = x * sin(v) + y * cos(v)
    return x_rot, y_rot, z

# Function to calculate initial conditions based on spherical direction
def compute_initial_conditions(polar_angle, azimuthal_angle):
    # Generate the initial direction vector in spherical coordinates.
    x, y, z = spherical_to_cartesian(1, polar_angle, azimuthal_angle)
    x, y, z = rotate_x(x, y, z)
    x, y, z = rotate_z(x, y, z)
    # Calculate velocities in spherical coordinates (r, θ, φ)
    x_vel = speed_of_light * x
    y_vel = speed_of_light * y
    z_vel = speed_of_light * z
    x = 2e11  # Shift x for the initial position
    y = 0
    z = 0
    # Apply inverse Boyer-Lindquist transformation
    boyer_lindquist_conditions = cartesian_to_boyer_lindquist(x, x_vel, y, y_vel, z, z_vel)
    
    # Format the conditions to [R, θ, φ, R_velocity, θ_velocity, φ_velocity].
    boyer_lindquist_conditions = np.array([boyer_lindquist_conditions[0],
                                             boyer_lindquist_conditions[2],
                                             boyer_lindquist_conditions[4],
                                             boyer_lindquist_conditions[1],
                                             boyer_lindquist_conditions[3],
                                             boyer_lindquist_conditions[5]])
    
    # Scale / adjust based on simulation unit conversion.
    boyer_lindquist_conditions = np.array([2 / r_s * boyer_lindquist_conditions[0],
                                             boyer_lindquist_conditions[1],
                                             boyer_lindquist_conditions[2],
                                             boyer_lindquist_conditions[3] / speed_of_light,
                                             boyer_lindquist_conditions[4] * r_s / (2 * speed_of_light),
                                             boyer_lindquist_conditions[5] * r_s / (2 * speed_of_light)])
    
    return boyer_lindquist_conditions

# --- Create the initial conditions array ---
# We now fill a grid over the two angular parameters.
init_conditions = np.zeros((n_rays, 6), dtype=np.double)

# Determine the theta and phi values.
# For example, polar angles could range from a small value up to pi/2.
theta_vals = np.linspace(0.1, np.pi, 10)  # Avoid theta = 0 to not run into singularities.
phi_vals = np.linspace(0, np.pi, n_phi, endpoint=False)
phi_vals = [np.pi / 3]

ray_index = 0
for theta in theta_vals:
    for phi in phi_vals:
        # Here, pass theta as the polar angle and phi as the azimuthal angle.
        init_conditions[ray_index] = compute_initial_conditions(theta, phi)
        ray_index += 1

# Flatten the initial conditions array to 1D for C compatibility.
init_conditions_flat = init_conditions.flatten()

# Allocate output arrays.
trajectories = np.zeros((n_rays, max_steps, state_dim), dtype=np.double)
steps_out = np.zeros(n_rays, dtype=np.int32)

# --- Load the DLL/shared library ---
import sys
if sys.platform.startswith('win'):
    lib = ctypes.CDLL('./bin/old_kerr.dll')
else:
    lib = ctypes.CDLL('./bin/libraykerr.so')

# Define argument types for the simulateRays function.
lib.simulateRays.argtypes = [ctypes.c_size_t,
                             ctypes.c_size_t,
                             np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')]

import time

# Start timing
start_time = time.time()

# Call the simulation
lib.simulateRays(n_rays, max_steps, init_conditions_flat, trajectories, steps_out)

# End timing
end_time = time.time()

# Print the duration
print(f"CUDA simulation completed in {end_time - start_time:.4f} seconds")

# --- Convert Boyer-Lindquist to Cartesian for 3D plotting ---
def boyer_lindquist_to_cartesian_3d(R, T, P, A_param):
    x = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.sin(P)
    z = R * np.cos(T)
    return x, y, z

# Recompute constants to ensure they match simulation parameters.
r_s = 2 * 6.67408e-11 * 8.543e36 / (299792458.0**2)
A = 0.99 * r_s / 2.0

# Create a 3D plot.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each ray trajectory (using valid steps as reported by steps_out)
for i in range(n_rays):
    n_steps = steps_out[i]
    if n_steps < 1:
        continue
    ray = trajectories[i, :n_steps, :]
    r_arr = ray[:, 0]
    theta_arr = ray[:, 1]
    phi_arr = ray[:, 2]
    # Convert from simulation (Boyer-Lindquist) coordinates to Cartesian (x, y, z)
    x, y, z = boyer_lindquist_to_cartesian_3d(r_s/2 * r_arr, theta_arr, phi_arr, A)
    ax.plot(x, y, z, lw=0.5, color='blue')

# Optionally, plot the event horizon as a 3D sphere.
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = r_s * np.outer(np.cos(u), np.sin(v))
y_sphere = r_s * np.outer(np.sin(u), np.sin(v))
z_sphere = r_s * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Ray Trajectories')

# Set plot limits.
ax.set_xlim(-r_env, r_env)
ax.set_ylim(-r_env, r_env)
ax.set_zlim(-r_env, r_env)

plt.show()