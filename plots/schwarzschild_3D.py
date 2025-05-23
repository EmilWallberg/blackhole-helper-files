import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

print("Code is live...")

rs = 1
r_0 = 20
du_0_values = np.linspace(-2, 2, 200)
phi_start = 0
phi_max = 3 * np.pi
h = 0.001
max_r = 30

def geodesic(u):
    return -u * (1 - (3 / 2) * rs * u)

def schwarzschild_func(phi, u):
    return [u[1], geodesic(u[0])]

def rk4_step(func, t, y, h):
    k1 = np.array(func(t, y))
    k2 = np.array(func(t + 0.5 * h, y + 0.5 * k1 * h))
    k3 = np.array(func(t + 0.5 * h, y + 0.5 * k2 * h))
    k4 = np.array(func(t + h, y + k3 * h))
    return y + (k1 + 2*k2 + 2*k3 + k4) * h / 6

def rotate_x(y, z, angle_rad):
    y_rot = y * np.cos(angle_rad) - z * np.sin(angle_rad)
    z_rot = y * np.sin(angle_rad) + z * np.cos(angle_rad)
    return y_rot, z_rot

print("Precomputing geodesics...")

# Precompute all geodesics (original 2D traces in x-z plane)
traces = []
u_0 = 1 / r_0
for du_0 in du_0_values:
    initial_conditions = np.array([u_0, du_0])
    phi_values = np.arange(phi_start, phi_max, h)
    u_du_values = np.zeros((len(phi_values), 2))
    u_du_values[0] = initial_conditions

    for i in range(1, len(phi_values)):
        phi = phi_values[i - 1]
        y = u_du_values[i - 1]
        if 1 / y[0] >= max_r:
            u_du_values[i - 1] = [1/max_r, 0]
            phi_values[i] = phi_values[i - 1]
            break
        u_du_values[i] = rk4_step(schwarzschild_func, phi, y, h)

    u_values = u_du_values[:, 0]
    r_values = 1 / u_values

    valid_indices = np.where(r_values >= rs)[0]
    invalid_mask = r_values < rs
    if np.any(invalid_mask):
        first_invalid_index = np.argmax(invalid_mask)
        valid_indices = valid_indices[valid_indices < first_invalid_index]

    r_values = r_values[valid_indices]
    phi_values = phi_values[valid_indices]

    x = r_values * np.cos(phi_values)
    z = r_values * np.sin(phi_values)
    y = np.zeros_like(x)

    traces.append((x, y, z))

print("Setting up figure...")

# Set up plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Geodesic motion in Schwarzschild spacetime (rotating around x-axis)')

# Draw central black hole as a sphere
u_sphere = np.linspace(0, 2 * np.pi, 50)
v_sphere = np.linspace(0, np.pi, 50)
U, V = np.meshgrid(u_sphere, v_sphere)
x_sphere = rs * np.cos(U) * np.sin(V)
y_sphere = rs * np.sin(U) * np.sin(V)
z_sphere = rs * np.cos(V)
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='black', alpha=1.0)

max_range = 35
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.grid(True)

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")

lines = [ax.plot([], [], [], color='blue', alpha=0.5)[0] for _ in traces]

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def update(frame):
    angle = 2 * np.pi * frame / 90  # full 360° in 90 frames
    for i, (x, y, z) in enumerate(traces):
        y_rot, z_rot = rotate_x(y, z, angle)
        lines[i].set_data(x, y_rot)
        lines[i].set_3d_properties(z_rot)
    return lines

anim = FuncAnimation(fig, update, frames=90, init_func=init, interval=50, blit=True)

plt.tight_layout()
plt.show()
