import ctypes
import numpy as np
import matplotlib.pyplot as plt

# Load CUDA shared library
cuda_lib = ctypes.CDLL("blackhole-helper-files/bin/rk4_schwarzchild.dll")

# Define argument types for the cuda_test function
cuda_lib.cuda_test.argtypes = [
    ctypes.c_double,                    # Black hole radius
    ctypes.c_double,                    # Environment map radius
    ctypes.c_int,                       # num_paths
    ctypes.c_int,                       # num_steps
    ctypes.c_double,                    # u_0
    ctypes.POINTER(ctypes.c_double),    # du_0_values (input array)
    ctypes.c_double,                    # h step size
    ctypes.c_double,                    # Tol tolerance 
    ctypes.POINTER(ctypes.c_double),    # u_out (output array for plot in py)
    ctypes.POINTER(ctypes.c_double),    # phi_out (output array for plot in py)
    ctypes.POINTER(ctypes.c_double),    # angle_out (output array)
]


num_steps = 500000
rs = 1.0
r = 20
u_0 = 1.0 / r
h = 0.001
tol = 0.001
envmap_r = 30.0


du_0_values = np.linspace(-2, 2, 100)
num_paths = len(du_0_values)

u_out = np.zeros(num_paths * num_steps, dtype=np.float64)
phi_out = np.zeros(num_paths * num_steps, dtype=np.float64)
angles_out = np.zeros(num_paths * 3, dtype=np.float64)


# Call the CUDA function
cuda_lib.cuda_test(rs, envmap_r, num_paths, num_steps, u_0,
                   du_0_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), h, tol,
                   u_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   phi_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                   angles_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                   )

# print("Hello World!")

end_angle_values = angles_out[1::3]
end_r_values = 1/angles_out[2::3]
start_angle = angles_out[0::3]

for i in range(len(start_angle)):
    print("[s: {:.10f}, e: {:.10f}]".format(start_angle[i], end_angle_values[i]))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(end_angle_values, end_r_values, color='blue', alpha=0.7)
axs[0].set_title('Scatter Plot of End Angles vs r Values')
axs[0].set_xlabel('End Angles')
axs[0].set_ylabel('End r Values')

# Add the circle to the plot representing the black hole
circle = plt.Circle((0, 0), rs, color='black', label='Central Mass', alpha=0.5)
axs[1].add_artist(circle)

u_out = u_out.reshape((num_paths, num_steps))
phi_out = phi_out.reshape((num_paths, num_steps))
r_out = 1 / u_out


for i in range(num_paths):
    r_values = r_out[i]

    r_values = r_out[i]
    phi_values = phi_out[i]

    x_values = r_values * np.cos(phi_values)
    y_values = r_values * np.sin(phi_values)

    axs[1].plot(x_values, y_values, label=f"Path {i}")

axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].set_title("Light Geodesics around a Black Hole")
axs[1].set_aspect('equal', adjustable='box')
axs[1].grid()

plt.tight_layout()
plt.show()
