import ctypes
import numpy as np
import matplotlib.pyplot as plt

from math import cos, sin, sqrt, atan2, acos

# Constants
speed_of_light = 299792458
M = 1
mass = 8.543e36

a = 0.99
Newton_G = 6.67408e-11
#r_s = 2 * Newton_G * mass / speed_of_light**2
r_s = 2 * 6.67408e-11 * 8.543e36 / (299792458.0**2)
A = a * r_s / 2  # Might be unit conversion form SI units to Natrual units
epsilon = 1e-10

r_env = 3.5e11
plot_scale = 1.1

print("Starting...")
# --- Parameters and settings ---
n_rays = 10
max_steps = 1000
state_dim = 5  # [r, theta, phi, p_r, p_theta]

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

# Allocate output arrays.
trajectories = np.zeros((n_rays * n_rays, max_steps, state_dim), dtype=np.float32)
steps_out = np.zeros(n_rays * n_rays, dtype=np.int32)

# --- Load the DLL/shared library ---
import sys
if sys.platform.startswith('win'):
    lib = ctypes.CDLL('./bin/kerr.dll')
else:
    lib = ctypes.CDLL('./bin/libraykerr.so')

lib.simulateRays.argtypes = [
    ctypes.c_float,  # x
    ctypes.c_float,  # y
    ctypes.c_float,  # z
    ctypes.c_float,  # rs (Schwarzschild radius)
    ctypes.c_float,  # Kerr parameter (a)
    ctypes.c_uint,   # num_rays_per_dim
    ctypes.c_uint,   # num_steps
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),  # trajectories_host
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')     # steps_out_host
]


# Call the simulation with different initial positions for each ray
pos =  [2e11, 2e11, 2e11]
lib.simulateRays(pos[0], pos[1], pos[2], r_s, a, n_rays, max_steps, trajectories, steps_out)

# --- Convert Boyer-Lindquist to Cartesian for plotting ---
def boyer_lindquist_to_cartesian(R, T, P, A_param):
    x = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.sin(P)
    z = R * np.cos(T)
    return x, y, z

# Create a 3D plot.
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot each ray trajectory (using valid steps as reported by steps_out)
for i in range(n_rays * n_rays):
    n_steps = steps_out[i]
    if n_steps < 1:
        continue
    ray = trajectories[i, :n_steps, :]
    r_arr = ray[:, 0]
    theta_arr = ray[:, 1]
    phi_arr = ray[:, 2]
    # Convert from simulation (Boyer-Lindquist) coordinates to Cartesian (x, y, z)
    x, y, z = boyer_lindquist_to_cartesian(r_s/2 * r_arr, theta_arr, phi_arr, A)
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
# Show the plot
plt.show()
print("Done.")
