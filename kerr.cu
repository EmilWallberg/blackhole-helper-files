#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Device constants (set at compile time; you may also update via
// cudaMemcpyToSymbol)
__constant__ double a = 0.99;
__constant__ double M = 1.0;         // Mass parameter
__constant__ double epsilon = 1e-10; // Numerical tolerance

// Additional simulation parameters
__constant__ double h_val = 0.1;      // Integration step size
__constant__ double r_env_val = 30.0; // Environment boundary

// ---------------------------------------------------------------------
// Kerr metric helper functions

__device__ double sigma(double r, double theta) {
  double cos_theta = cos(theta);
  return r * r + a * a * cos_theta * cos_theta;
}

__device__ double delta_r(double r) { return r * r + a * a - 2.0 * M * r; }

__device__ double ddelta_r(double r) { return 2.0 * r - 2.0 * M; }

// ---------------------------------------------------------------------
// Functions W_r, W_theta and their derivatives

__device__ double W_r(double r, double E, double L) {
  return E * (r * r + a * a) - a * L;
}

__device__ double dWsquare_r(double r, double E, double L) {
  double W = W_r(r, E, L);
  double dW_dr = 2.0 * E * r;
  return 2.0 * W * dW_dr;
}

__device__ double W_theta(double theta, double E, double L) {
  double sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, epsilon);
  return a * E * sin_theta - L / sin_theta;
}

__device__ double dWsquare_theta(double theta, double E, double L) {
  double sin_theta = sin(theta);
  double cos_theta = cos(theta);
  sin_theta = fmax(sin_theta, epsilon);
  double dW_dtheta = cos_theta * (a * E + L / (sin_theta * sin_theta));
  return 2.0 * W_theta(theta, E, L) * dW_dtheta;
}

// ---------------------------------------------------------------------
// Definitions of the conserved quantities and derived functions

__device__ double E_func(double r, double theta, double dr, double dtheta,
                         double dphi) {
  double sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, epsilon);
  double delta = delta_r(r);
  double term = ((a * a * sin_theta * sin_theta - delta) *
                     (-dr * dr / delta - dtheta * dtheta) +
                 (dphi * sin_theta) * (dphi * sin_theta) * delta);
  return sqrt(term);
}

__device__ double L_func(double r, double theta, double dphi, double E) {
  double sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, epsilon);
  double delta = delta_r(r);
  double sigma_val = sigma(r, theta);
  double num =
      a * E * delta + (sigma_val * delta * dphi - a * E * (r * r + a * a));
  double denom = delta - a * a * sin_theta * sin_theta;
  return sin_theta * sin_theta * num / denom;
}

__device__ double k_func(double r, double theta, double dr, double E,
                         double L) {
  double sigma_val = sigma(r, theta);
  double delta = delta_r(r);
  double W = W_r(r, E, L);
  return (W * W - sigma_val * sigma_val * dr * dr) / delta;
}

// ---------------------------------------------------------------------
// Geodesic equations: state vector y = [r, theta, phi, p_r, p_theta]

__device__ double dr_func(double r, double theta, double p_r) {
  return delta_r(r) * p_r / sigma(r, theta);
}

__device__ double dtheta_func(double r, double theta, double p_theta) {
  return p_theta / sigma(r, theta);
}

__device__ double dphi_func(double r, double theta, double E, double L) {
  double sig = sigma(r, theta);
  double delta = delta_r(r);
  double sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, epsilon);
  return (a * W_r(r, E, L) / delta - W_theta(theta, E, L) / sin_theta) / sig;
}

__device__ double dp_r(double r, double theta, double p_r, double E, double L,
                       double k_val) {
  double sig = sigma(r, theta);
  double delta = delta_r(r);
  double d_delta = ddelta_r(r);
  double dW2 = dWsquare_r(r, E, L);
  double num = dW2 - d_delta * k_val;
  return (num / (2.0 * delta) - d_delta * p_r * p_r) / sig;
}

__device__ double dp_theta(double r, double theta, double E, double L) {
  double sig = sigma(r, theta);
  double dW_theta_val = dWsquare_theta(theta, E, L);
  return -dW_theta_val / (2.0 * sig);
}

// ---------------------------------------------------------------------
// RK4 integration using a loop to compute k coefficients
// The state vector y has 5 components.
__device__ void rk4(double *y, double h, double E, double L, double k_val) {
  double k[4][5];   // k coefficients for the 4 stages
  double y_temp[5]; // temporary storage

  // Loop over the 4 stages
  for (int stage = 0; stage < 4; ++stage) {
    double factor = (stage == 0) ? 0.0 : (stage == 3 ? 1.0 : 0.5);
    // Compute temporary state: y_temp = y + factor * h * (previous k)
    // For stage 0 we simply have y_temp = y.
    for (int i = 0; i < 5; ++i)
      y_temp[i] = y[i] + (stage == 0 ? 0.0 : factor * h * k[stage - 1][i]);

    // Compute the derivatives at y_temp
    k[stage][0] = dr_func(y_temp[0], y_temp[1], y_temp[3]);
    k[stage][1] = dtheta_func(y_temp[0], y_temp[1], y_temp[4]);
    k[stage][2] = dphi_func(y_temp[0], y_temp[1], E, L);
    k[stage][3] = dp_r(y_temp[0], y_temp[1], y_temp[3], E, L, k_val);
    k[stage][4] = dp_theta(y_temp[0], y_temp[1], E, L);
  }
  // Combine the stages
  for (int i = 0; i < 5; ++i) {
    y[i] += h / 6.0 * (k[0][i] + 2.0 * k[1][i] + 2.0 * k[2][i] + k[3][i]);
  }
}

// ---------------------------------------------------------------------
// Kernel: each thread simulates one ray.
// Input initial conditions are in the order:
// [r0, theta0, phi0, dr0, dtheta0, dphi0]
// The output trajectory (state vector per step) and the number of steps per ray
// are stored in contiguous device memory.
__global__ void simulateRayKernel(int num_rays, const double *init_conditions,
                                  double *trajectories, int *steps_out,
                                  int num_steps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rays)
    return;

  // Load initial conditions for the idx-th ray.
  const double *init = &init_conditions[idx * 6];
  double r0 = init[0];
  double theta0 = init[1];
  double phi0 = init[2];
  double dr0 = init[3];
  double dtheta0 = init[4];
  double dphi0 = init[5];

  // Compute conserved quantities using Kerr equations.
  double E = E_func(r0, theta0, dr0, dtheta0, dphi0);
  double L = L_func(r0, theta0, dphi0, E);
  double k_val = k_func(r0, theta0, dr0, E, L);

  // Compute initial momenta.
  double S = sigma(r0, theta0);
  double p_r0 = S * dr0 / delta_r(r0);
  double p_theta0 = S * dtheta0;

  // Set up the initial state vector: [r, theta, phi, p_r, p_theta]
  double y[5];
  y[0] = r0;
  y[1] = theta0;
  y[2] = phi0;
  y[3] = p_r0;
  y[4] = p_theta0;

  // Pointer to this ray's trajectory data.
  double *ray_traj = &trajectories[idx * num_steps * 5];

  int step;
  for (step = 0; step < num_steps; step++) {
    // Terminate integration if ray is inside the horizon or outside the
    // environment.
    if (y[0] < 1.99)
      break;

    // Store the current state into the trajectory.
    for (int j = 0; j < 5; j++) {
      ray_traj[step * 5 + j] = y[j];
    }
    // Advance one RK4 step.
    rk4(y, h_val, E, L, k_val);
  }
  steps_out[idx] = step;
}

// ---------------------------------------------------------------------
// Exported function for DLL interface
// This function is called from Python via a DLL (or shared library).
// It accepts the number of rays, number of integration steps, and an array
// of initial conditions (size: num_rays * 6). It outputs the trajectory data
// (num_rays * num_steps * 5 double values) and the number of steps for each
// ray.
extern "C" {

#ifdef _WIN32
__declspec(dllexport)
#endif
void simulateRays(size_t num_rays, size_t num_steps, const double *init_conditions_host, double *trajectories_host, int *steps_out_host, const double a_black_hole) {
  // Calculate sizes for memory allocation.
  size_t init_size = num_rays * 6 * sizeof(double);
  size_t traj_size = num_rays * num_steps * 5 * sizeof(double);
  size_t steps_size = num_rays * sizeof(int);

  cudaMemcpyToSymbol(a, &a_black_hole, sizeof(double));

  // Allocate device memory.
  double *d_init_conditions = nullptr;
  double *d_trajectories = nullptr;
  int *d_steps_out = nullptr;
  cudaMalloc(&d_init_conditions, init_size);
  cudaMalloc(&d_trajectories, traj_size);
  cudaMalloc(&d_steps_out, steps_size);

  // Copy initial conditions from host to device.
  cudaMemcpy(d_init_conditions, init_conditions_host, init_size,
             cudaMemcpyHostToDevice);

  // Determine kernel launch configuration.
  int threadsPerBlock = 256;
  int blocks = (int)((num_rays + threadsPerBlock - 1) / threadsPerBlock);

  // Launch the simulation kernel.
  simulateRayKernel<<<blocks, threadsPerBlock>>>(
      (int)num_rays, d_init_conditions, d_trajectories, d_steps_out,
      (int)num_steps);
  cudaDeviceSynchronize();

  // Copy the results back to host.
  cudaMemcpy(trajectories_host, d_trajectories, traj_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(steps_out_host, d_steps_out, steps_size, cudaMemcpyDeviceToHost);

  // Free device memory.
  cudaFree(d_init_conditions);
  cudaFree(d_trajectories);
  cudaFree(d_steps_out);
}

} // extern "C"