import ctypes
import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, sqrt, atan2, acos

# Constants
speed_of_light = 299792458
M = 1
mass = 8.543e36

a = 0.6
Newton_G = 6.67408e-11
r_s = 2 * Newton_G * mass / speed_of_light**2
A = a * r_s / 2 # Might be unit conversion form SI units to Natrual units
epsilon = 1e-10

r_env = 3.5e11
plot_scale = 1.1


# --- Parameters and settings ---
n_rays = 200
max_steps = 1000
state_dim = 5   # [r, theta, phi, p_r, p_theta]

import numpy as np

# Spherical to Cartesian conversion (r, θ, φ -> x, y, z)
def spherical_to_cartesian(radius, polar_angle, azimuthal_angle):
    x = radius * np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = radius * np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = radius * np.cos(polar_angle)
    return np.array([x, y, z])

# Boyer-Lindquist coordinates to Cartesian (R, θ, φ -> x, y, z)
def boyer_lindquist_to_cartesian(R,T,P):
    x = np.sqrt(R**2+A**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2+A**2)* np.sin(T) * np.sin(P)
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
    radius_velocity = (radius * (x * x_velocity + y * y_velocity + z * z_velocity)) / (2 * radius**2 + A**2 - x**2 - y**2 - z**2) + \
                      A**2 * z * z_velocity / (radius * (2 * radius**2 + A**2 - x**2 - y**2 - z**2))
    polar_angle_velocity = (z * radius_velocity - z_velocity * radius) / (radius * sqrt(radius**2 - z**2))
    azimuthal_angle_velocity = (y_velocity * x - x_velocity * y) / (x**2 + y**2)
    
    return np.array([radius, radius_velocity, polar_angle, polar_angle_velocity, azimuthal_angle, azimuthal_angle_velocity])

def rotate_x(x, y, z, v  = np.pi/2):
    # 90 degrees around x-axis
    y_rot = y * cos(v) - z * sin(v)
    z_rot = y * sin(v) + z * cos(v)
    return x, y_rot, z_rot

def rotate_z(x, y, z, v = np.pi / 2):
    # -90 degrees around z-axis
    x_rot = x * cos(v) - y * sin(v)
    y_rot = x * sin(v) + y * cos(v)
    return x_rot, y_rot, z

# Function to calculate initial conditions based on spherical direction
def compute_initial_conditions(polar_angle, azimuthal_angle):
    x, y, z = spherical_to_cartesian(1, polar_angle, azimuthal_angle)
    x, y, z = rotate_x(x, y, z)
    x, y, z = rotate_z(x, y, z)
    # Calculate velocities in spherical coordinates (r, θ, φ)
    x_vel = speed_of_light * x
    y_vel = speed_of_light * y
    z_vel = speed_of_light * z
    x += 2e11
    
    # Apply inverse Boyer-Lindquist transformation
    boyer_lindquist_conditions = cartesian_to_boyer_lindquist(x, x_vel, y, y_vel,
                                                              z, z_vel)
    
    # Adjust the output to the required format (R, θ, φ, velocities)
    boyer_lindquist_conditions = np.array([boyer_lindquist_conditions[0], boyer_lindquist_conditions[2], boyer_lindquist_conditions[4], 
                                           boyer_lindquist_conditions[1], boyer_lindquist_conditions[3], boyer_lindquist_conditions[5]])
    boyer_lindquist_conditions = np.array([2 / r_s * boyer_lindquist_conditions[0], boyer_lindquist_conditions[1], boyer_lindquist_conditions[2],
                                           boyer_lindquist_conditions[3] / speed_of_light, boyer_lindquist_conditions[4] * r_s / (2 * speed_of_light),
                                           boyer_lindquist_conditions[5] * r_s / (2 * speed_of_light)])
    
    return boyer_lindquist_conditions

# Example: create initial conditions array [n_rays x 6]
# For demonstration, we create simple initial conditions.
# In practice, use your compute_initial_conditions() to get proper values.
init_conditions = np.zeros((n_rays, 6), dtype=np.double)
angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
for i, angle in enumerate(angles):
    # Use azimuthal angle = angle, and fix polar angle = 0 for 2D rays
    init_conditions[i] = compute_initial_conditions(angle, 0)

# Flatten the array for C compatibility.
init_conditions_flat = init_conditions.flatten()

# Allocate output arrays.
trajectories = np.zeros((n_rays, max_steps, state_dim), dtype=np.double)
steps_out = np.zeros(n_rays, dtype=np.int32)

# --- Load the DLL/shared library ---
import sys
if sys.platform.startswith('win'):
    lib = ctypes.CDLL('./bin/kerr.dll')
else:
    lib = ctypes.CDLL('./bin/libraykerr.so')

# Define argument types for the simulateRays function.
# simulateRays(int n_rays, const double *init_conditions, double *trajectories, int *steps_out)
lib.simulateRays.argtypes = [ctypes.c_size_t,
                             ctypes.c_size_t,
                             np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
                             np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
                            ctypes.c_double]


# Call the simulation.
lib.simulateRays(n_rays, max_steps, init_conditions_flat, trajectories, steps_out, a)

# --- Convert Boyer-Lindquist to Cartesian for plotting ---
def boyer_lindquist_to_cartesian(R, T, P, A_param):
    # Following your python conversion: 
    x = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.sin(P)
    z = R * np.cos(T)
    return x, y, z

# Use constant r_s (same as in the CUDA code) and A.
# (Make sure these match the ones in your simulation.)
# r_s = 2 * 6.67408e-11 * 8.543e36 / (299792458.0**2)
# A = 0.99 * r_s / 2.0

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# Plot each ray trajectory (only use valid steps as reported by steps_out)
# Plot each ray trajectory (only use valid steps as reported by steps_out)
for i in range(n_rays):
    n_steps = steps_out[i]
    if n_steps < 1:
        continue
    ray = trajectories[i, :n_steps, :]
    r_arr, theta_arr, phi_arr = ray[:, 0], ray[:, 1], ray[:, 2]
    
    # Filter out steps where the radius exceeds r_env
    r_phys = (r_s / 2) * r_arr
    valid_steps = (r_phys <= r_env) & (r_phys > r_s)


    # If there are no valid steps, skip plotting this ray
    if np.any(valid_steps):
        r_arr = r_arr[valid_steps]
        theta_arr = theta_arr[valid_steps]
        phi_arr = phi_arr[valid_steps]
        
        # Convert to Cartesian coordinates (for 2D plotting, we use x and y)
        x, y, _ = boyer_lindquist_to_cartesian(r_s/2 * r_arr, theta_arr, phi_arr, A)
        ax.plot(x, y, lw=0.5, color='blue')


# Draw event horizon circle (2D)
theta_circle = np.linspace(0, 2*np.pi, 100)
x_circle = r_s * np.cos(theta_circle)
y_circle = r_s * np.sin(theta_circle)
ax.fill(x_circle, y_circle, color='black')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(-plot_scale * r_env, plot_scale * r_env)
ax.set_ylim(-plot_scale * r_env, plot_scale * r_env)
plt.show()