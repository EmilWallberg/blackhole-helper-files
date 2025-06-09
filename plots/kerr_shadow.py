import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# --- Physical Parameters ---
m = 4.1e6
a = 0.99 * m            # spin
r_o = 5 * m            # observer radius
theta_o = np.pi / 3    # general inclination (π/2 = equator)
l = 0
Lambda = 0

# --- Metric Functions ---
def delta_r(r): return r**2 - 2*m*r + a**2
def d_delta_r(r): return 2*r - 2*m

def K_E(rp):  # Carter constant
    num = 16 * rp**2 * delta_r(rp)
    denom = d_delta_r(rp)**2
    return num / denom if denom != 0 else np.nan

def L_tilde(rp):  # Reduced angular momentum
    denom = a if a != 0 else 1e-10
    return (rp**2 + l**2 - (4 * rp * delta_r(rp)) / d_delta_r(rp)) / denom

def psi_func(Ltilde, KE):
    denom = np.sqrt(KE) * np.sin(theta_o)
    if denom == 0: return np.nan
    arg = (Ltilde + a * np.cos(theta_o)**2 + 2 * l * np.cos(theta_o)) / denom
    return np.arcsin(np.clip(arg, -1, 1))

def theta_func(KE, Ltilde):
    num = np.sqrt(delta_r(r_o) * KE)
    denom = r_o**2 + l**2 - a * Ltilde
    if denom == 0: return np.nan
    arg = num / denom
    return np.arcsin(np.clip(arg, 0, 1))

# --- Solve geodesic equation per ψ (from Eq. 27) ---
def solve_shadow():
    r_plus = m + np.sqrt(m**2 - a**2)
    r_vals = np.linspace(r_plus + 1e-4, 6 * m, 800)  # Parameter: radius of photon orbits
    x_vals, y_vals = [], []

    for rp in r_vals:
        if delta_r(rp) <= 0 or d_delta_r(rp) == 0:
            continue

        KE = K_E(rp)
        Ltilde = L_tilde(rp)
        if KE <= 0 or np.isnan(KE): continue

        psi = psi_func(Ltilde, KE)
        theta = theta_func(KE, Ltilde)
        if np.isnan(theta) or np.isnan(psi): continue

        x = -2 * np.tan(theta / 2) * np.sin(psi)
        y = -2 * np.tan(theta / 2) * np.cos(psi)
        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)

# --- Schwarzschild special case ---
def schwarzschild_shadow():
    rho_o = r_o / (2 * m)
    sin2_alpha = (27 / 4) * ((rho_o - 1) / rho_o**3)
    alpha = np.arcsin(np.sqrt(np.clip(sin2_alpha, 0, 1)))
    R = 2 * np.tan(alpha / 2)
    t = np.linspace(0, 2*np.pi, 500)
    x = R * np.cos(t)
    y = R * np.sin(t)
    return x, y

# --- Main ---
if a == 0:
    x_vals, y_vals = schwarzschild_shadow()
else:
    x_vals, y_vals = solve_shadow()

# --- Plot ---
plt.figure(figsize=(7, 7))
plt.plot(x_vals, y_vals, color='black', label=f'a = {a/m:.2f} m')
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Black Hole Shadow\nObserver at r = {r_o/m:.1f} m, θ = {theta_o*180/np.pi:.1f}°")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()