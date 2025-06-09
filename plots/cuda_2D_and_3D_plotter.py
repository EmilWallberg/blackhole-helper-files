import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Constants
speed_of_light = 299792458
mass = 8.543e36
a = 0.99
Newton_G = 6.67408e-11
r_s = 2 * Newton_G * mass / speed_of_light**2
A = a * r_s / 2  # Kerr parameter scaled
epsilon = 1e-10

r_env = 3.5e11

print("Starting...")

# Parameters
n_rays = 50
max_steps = 1000
state_dim = 5  # [r, theta, phi, p_r, p_theta]

# Set True to get 3D rotated rays, False for simple 2D plot in equatorial plane
plot_3d = True

# Boyer-Lindquist to Cartesian conversion
def boyer_lindquist_to_cartesian(R, T, P, A_param):
    x = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.cos(P)
    y = np.sqrt(R**2 + A_param**2) * np.sin(T) * np.sin(P)
    z = R * np.cos(T)
    return x, y, z

# Rotation of trajectory points around x-axis by angle_rad
def rotate_around_x(x, y, z, angle_rad):
    y_rot = y * np.cos(angle_rad) - z * np.sin(angle_rad)
    z_rot = y * np.sin(angle_rad) + z * np.cos(angle_rad)
    return x, y_rot, z_rot

# Allocate output arrays
trajectories = np.zeros((n_rays * n_rays, max_steps, state_dim), dtype=np.float32)
steps_out = np.zeros(n_rays * n_rays, dtype=np.int32)

# Load DLL/shared library
import sys
if sys.platform.startswith('win'):
    lib = ctypes.CDLL('./bin/kerr.dll')
else:
    lib = ctypes.CDLL('./bin/libraykerr.so')

lib.simulateRays.argtypes = [
    ctypes.c_float,  # x
    ctypes.c_float,  # y
    ctypes.c_float,  # z
    ctypes.c_float,  # rs
    ctypes.c_float,  # Kerr parameter (a)
    ctypes.c_uint,   # num_rays_per_dim
    ctypes.c_uint,   # num_steps
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
]

# Initial position of rays starting at x=2e11, y=0, z=0
pos = [2e11, 0, 0]
lib.simulateRays(pos[0], pos[1], pos[2], r_s, a, n_rays, max_steps, trajectories, steps_out)

# Setup 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

if plot_3d:
    # Number of rotation steps around x-axis to create 3D effect
    n_rotations = 15
    angles = np.linspace(0, np.pi, n_rotations)  # rotate from 0 to 180 degrees

    # Plot rays multiple times rotated around x-axis
    for angle in angles:
        for i in range(n_rays * n_rays):
            n_steps = steps_out[i]
            if n_steps < 1:
                continue
            ray = trajectories[i, :n_steps, :]
            r_arr = ray[:, 0]
            theta_arr = ray[:, 1]
            phi_arr = ray[:, 2]

            # Convert from Boyer-Lindquist to Cartesian coordinates
            x, y, z = boyer_lindquist_to_cartesian(r_s/2 * r_arr, theta_arr, phi_arr, A)

            # Rotate trajectory around x-axis by angle
            x_r, y_r, z_r = rotate_around_x(x, y, z, angle)
            ax.plot(x_r, y_r, z_r, lw=0.6, alpha=0.5, color='blue')
else:
    angle = np.pi / 2  # rotate xy-plane rays into xz-plane
    for i in range(n_rays * n_rays):
        n_steps = steps_out[i]
        if n_steps < 1:
            continue
        ray = trajectories[i, :n_steps, :]
        r_arr = ray[:, 0]
        theta_arr = ray[:, 1]
        phi_arr = ray[:, 2]

        x, y, z = boyer_lindquist_to_cartesian(r_s/2 * r_arr, theta_arr, phi_arr, A)
        x, y, z = rotate_around_x(x, y, z, angle)
        ax.plot(x, y, z, lw=0.6, alpha=0.7, color='blue')

# Plot event horizon sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = r_s * np.outer(np.cos(u), np.sin(v))
y_sphere = r_s * np.outer(np.sin(u), np.sin(v))
z_sphere = r_s * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=0.3)

# Labels and limits
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Ray Trajectories' + (" (Rotated 3D)" if plot_3d else " (Equatorial Plane)"))

max_range = r_env * 1.5
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=30, azim=45)

plt.show()
print("Done.")
