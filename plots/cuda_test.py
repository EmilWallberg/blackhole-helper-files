import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load CUDA shared library
cuda_lib = ctypes.CDLL("blackhole-helper-files/bin/rk4_schwarzchild.dll")

# Define argument types for the cuda_test function
cuda_lib.cuda_test.argtypes = [
    ctypes.c_int,  # num_paths
    ctypes.c_int,  # num_steps
    ctypes.c_double,  # u_0
    ctypes.POINTER(ctypes.c_double),  # du_0_values (input array)
    ctypes.c_double, # h step size
    ctypes.POINTER(ctypes.c_double),  # u_out (output array)
    ctypes.POINTER(ctypes.c_double),  # phi_out (output array)
    ctypes.POINTER(ctypes.c_double),
]

# Prepare data to pass to CUDA function
num_steps = 5000
r = 20
u_0 = 1.0 / r
h = 0.01

# Initial du_0 values as a numpy array (double precision)
du_0_values = np.linspace(-2, 2, 100)
num_paths = len(du_0_values)

# Prepare output arrays to hold results
u_out = np.zeros(num_paths * num_steps, dtype=np.float64)
phi_out = np.zeros(num_paths * num_steps, dtype=np.float64)
angles_out = np.zeros(num_paths * 3, dtype=np.float64)


# Call the CUDA function
cuda_lib.cuda_test(num_paths, num_steps, u_0,
                   du_0_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), h,
                   u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   angles_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                   )

# Select every third value starting from index 2 for angles_out
end_angle_values = angles_out[1::3]

# Select every third value starting from index 3 for u_out
end_r_values = 1/angles_out[2::3]

# Create a figure with 1 row and 2 columns of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scatter plot on the first subplot (end_angle_values vs end_r_values)
axs[0].scatter(end_angle_values, end_r_values, color='blue', alpha=0.7)
axs[0].set_title('Scatter Plot of End Angles vs r Values')
axs[0].set_xlabel('End Angles')
axs[0].set_ylabel('End r Values')

# Plot light geodesics around a black hole on the second subplot
rs = 1  # Radius of central mass (black hole)
circle = plt.Circle((0, 0), rs, color='black', label='Central Mass', alpha=0.5)

# Reshape u_out and phi_out for plotting
u_out = u_out.reshape((num_paths, num_steps))
phi_out = phi_out.reshape((num_paths, num_steps))
r_out = 1 / u_out  # Convert u to r

# Add the circle to the plot
axs[1].add_artist(circle)

for i in range(num_paths):
    r_values = r_out[i]

    r_values = r_out[i]
    phi_values = phi_out[i]

    x_values = r_values * np.cos(phi_values)
    y_values = r_values * np.sin(phi_values)

    # Plot results on the second subplot
    axs[1].plot(x_values, y_values, label=f"Path {i}")

# Set axis labels and title for the second subplot
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
axs[1].set_title("Light Geodesics around a Black Hole")
axs[1].set_aspect('equal', adjustable='box')
axs[1].grid()

plt.tight_layout()
plt.show()
