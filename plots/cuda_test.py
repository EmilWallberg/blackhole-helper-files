import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load CUDA shared library
cuda_lib = ctypes.CDLL("../rk4_schwarzchild.cu")

# Define argument types for the cuda_test function
cuda_lib.cuda_test.argtypes = [
    ctypes.c_int,  # num_paths
    ctypes.c_int,  # num_steps
    ctypes.c_double,  # u_0
    ctypes.POINTER(ctypes.c_double),  # du_0_values (input array)
    ctypes.c_double, # h step size
    ctypes.POINTER(ctypes.c_double),  # u_out (output array)
    ctypes.POINTER(ctypes.c_double),  # phi_out (output array)
]

# Prepare data to pass to CUDA function
num_steps = 5000
r = 200
u_0 = 1.0 / r
h = 0.001

# Initial du_0 values as a numpy array (double precision)
du_0_values = np.linspace(-2, 2, 1000)
num_paths = len(du_0_values)

# Prepare output arrays to hold results
u_out = np.zeros(num_paths * num_steps, dtype=np.float64)
phi_out = np.zeros(num_paths * num_steps, dtype=np.float64)

# Call the CUDA function
cuda_lib.cuda_test(num_paths, num_steps, u_0,
                   du_0_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), h,
                   u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

plt.figure(figsize=(8, 8))
rs = 1
# Create a circle with the desired radius, center at (0, 0), and filled in
circle = plt.Circle((0, 0), rs, color='black', label='Central Mass')

# Add the circle to the plot
plt.gca().add_artist(circle)

# Assuming u_out and phi_out are already reshaped (num_paths, num_steps)
# and you have r_out as well
u_out = u_out.reshape((num_paths, num_steps))
phi_out = phi_out.reshape((num_paths, num_steps))
r_out = 1 / u_out  # Convert u to r

for i in range(num_paths):
    r_values = r_out[i]
    valid_indices = np.where(r_values >= 1)[0]

   # Find the first index where the condition does not hold
    first_invalid_index = np.argmax(r_values < 1)
    valid_indices = valid_indices[valid_indices < first_invalid_index]
        
    r_values = r_out[i][valid_indices]
    phi_values = phi_out[i][valid_indices]

# Convert to Cartesian coordinates
# r_out and phi_out should be reshaped to match the paths and steps
    x_values = r_values * np.cos(phi_values)
    y_values = r_values * np.sin(phi_values)

# Plot results
    plt.plot(x_values, y_values, label=f"Path {i}")

plt.xlabel("X")
plt.ylabel("Y")

# Set x and y axis limits
plt.xlim(-10, 30)
plt.ylim(-20, 20)

plt.grid()

plt.title("Light Geodesics around a Black Hole in Cartesian Coordinates")
plt.show()