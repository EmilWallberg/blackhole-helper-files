#include "device_launch_parameters.h"
#include "vector_functions.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifndef M_C
#define M_C 299792458.0f
#endif

// ---------------------------------------------------------------------
// Device constants (set at compile time; you may also update via
// cudaMemcpyToSymbol)
// Device constants (set at compile time; you may also update via
// cudaMemcpyToSymbol)
__constant__ float c_a = 0.6f;
__constant__ float c_rs = 1;
__constant__ unsigned int c_num_steps = 15000;
__constant__ unsigned int c_layers = 1;
__constant__ float c_M = 1.0f;        // Mass parameter
__constant__ float c_epsilon = 1e-10; // Numerical tolerance
__constant__ float3 worldUp = {0.0f, 1.0f, 0.0f};

// Additional simulation parameters
__constant__ float c_h = 0.1f;           // Integration step size
__constant__ double r_env_val = 1000.0f; // Environment boundary

// ---------------------------------------------------------------------
// Coordinate convertion functions

// helper math (as before)
__device__ float3 crossf3(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float3 normalizef3(const float3 &v) {
  float len2 = v.x * v.x + v.y * v.y + v.z * v.z;
  float invLen = 1.0f / sqrtf(len2);
  return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ inline float3 operator*(const float3 &v, float s) {
  return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float3 operator*(float s, const float3 &v) {
  return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 spherical_to_cartesian(float r, float theta, float phi) {
  return make_float3(r * sinf(theta) * cosf(phi), r * sinf(theta) * sinf(phi),
                     r * cosf(theta));
}

__device__ void cartesian_to_boyer_lindquist(float x, float x_vel, float y,
                                             float y_vel, float z, float z_vel,
                                             float A, float *out) {
  double r2 = x * x + y * y + z * z;
  double A2 = A * A;
  double root = sqrt(A2 * (A2 - 2.0 * (x * x + y * y) + 2.0 * z * z) + r2 * r2);
  double radius = sqrt((-A2 + r2 + root) * 0.5);

  float azimuthal_angle = atan2f(y, x);
  float polar_angle = acosf(z / radius);

  double denom = 2.0 * radius * radius + A2 - r2;
  double radius_velocity =
      ((radius * (x * x_vel + y * y_vel + z * z_vel)) / denom +
       A2 * z * z_vel / (radius * denom));

  float polar_denom = radius * sqrtf(radius * radius - z * z);
  float polar_velocity = (z * radius_velocity - z_vel * radius) / polar_denom;

  float azimuthal_velocity = (y_vel * x - x_vel * y) / (x * x + y * y);

  out[0] = radius;
  out[1] = radius_velocity;
  out[2] = polar_angle;
  out[3] = polar_velocity;
  out[4] = azimuthal_angle;
  out[5] = azimuthal_velocity;
}

// ---------------------------------------------------------------------
// Kerr metric helper functions

__device__ float sigma(float r, float theta) {
  float cos_theta = cos(theta);
  return r * r + c_a * c_a * cos_theta * cos_theta;
}

__device__ float delta_r(float r) { return r * r + c_a * c_a - 2.0f * c_M * r; }

__device__ float ddelta_r(float r) { return 2.0f * (r - c_M); }

// ---------------------------------------------------------------------
// Functions W_r, W_theta and their derivatives

__device__ float W_r(float r, float E, float L) {
  return E * (r * r + c_a * c_a) - c_a * L;
}

__device__ float dWsquare_r(float r, float E, float L) {
  float W = W_r(r, E, L);
  float dW_dr = 2.0f * E * r;
  return 2.0f * W * dW_dr;
}

__device__ float W_theta(float theta, float E, float L) {
  float sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, c_epsilon);
  return c_a * E * sin_theta - L / sin_theta;
}

__device__ float dWsquare_theta(float theta, float E, float L) {
  float sin_theta = sin(theta);
  float cos_theta = cos(theta);
  sin_theta = fmax(sin_theta, c_epsilon);
  float dW_dtheta = cos_theta * (c_a * E + L / (sin_theta * sin_theta));
  return 2.0f * W_theta(theta, E, L) * dW_dtheta;
}

// ---------------------------------------------------------------------
// Definitions of the conserved quantities and derived functions

__device__ float E_func(float r, float theta, float dr, float dtheta,
                        float dphi) {
  float sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, c_epsilon);
  float delta = delta_r(r);
  float term = ((c_a * c_a * sin_theta * sin_theta - delta) *
                    (-dr * dr / delta - dtheta * dtheta) +
                (dphi * sin_theta) * (dphi * sin_theta) * delta);
  return sqrt(term);
}

__device__ float L_func(float r, float theta, float dphi, float E) {
  float sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, c_epsilon);
  float delta = delta_r(r);
  float sigma_val = sigma(r, theta);
  float num = c_a * E * delta +
              (sigma_val * delta * dphi - c_a * E * (r * r + c_a * c_a));
  float denom = delta - c_a * c_a * sin_theta * sin_theta;
  return sin_theta * sin_theta * num / denom;
}

__device__ float k_func(float r, float theta, float dr, float E, float L) {
  float sigma_val = sigma(r, theta);
  float delta = delta_r(r);
  float W = W_r(r, E, L);
  return (W * W - sigma_val * sigma_val * dr * dr) / delta;
}

// ---------------------------------------------------------------------
// Geodesic equations: state vector y = [r, theta, phi, p_r, p_theta]
__device__ float dr_func(float r, float theta, float p_r) {
  return delta_r(r) * p_r / sigma(r, theta);
}

__device__ float dtheta_func(float r, float theta, float p_theta) {
  return p_theta / sigma(r, theta);
}

__device__ float dphi_func(float r, float theta, float E, float L) {
  float sig = sigma(r, theta);
  float delta = delta_r(r);
  float sin_theta = sin(theta);
  sin_theta = fmax(sin_theta, c_epsilon);
  return (c_a * W_r(r, E, L) / delta - W_theta(theta, E, L) / sin_theta) / sig;
}

__device__ float dp_r(float r, float theta, float p_r, float E, float L,
                      float k_val) {
  float sig = sigma(r, theta);
  float delta = delta_r(r);
  float d_delta = ddelta_r(r);
  float dW2 = dWsquare_r(r, E, L);
  float num = dW2 - d_delta * k_val;
  return (num / (2.0f * delta) - d_delta * p_r * p_r) / sig;
}

__device__ float dp_theta(float r, float theta, float E, float L) {
  float sig = sigma(r, theta);
  float dW_theta_val = dWsquare_theta(theta, E, L);
  return -dW_theta_val / (2.0f * sig);
}

__device__ float Ke(float k, float E) { return k / (E * E); }

__device__ float aLE(float L, float E) { return c_a * L / E; }

// ---------------------------------------------------------------------
// RK4 integration using c_a loop to compute k coefficients
// The state vector y has 5 components.
__device__ void rk4(float *y, float h, float E, float L, float k_val) {
  float k[4][5];   // k coefficients for the 4 stages
  float y_temp[5]; // temporary storage

  // Loop over the 4 stages
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    float factor = (stage == 0) ? 0.0f : (stage == 3 ? 1.0f : 0.5f);
    // Compute temporary state: y_temp = y + factor * h * (previous k)
    // For stage 0 we simply have y_temp = y.
#pragma unroll
    for (int i = 0; i < 5; ++i)
      y_temp[i] = y[i] + (stage == 0 ? 0.0f : factor * h * k[stage - 1][i]);

    // Compute the derivatives at y_temp
    k[stage][0] = dr_func(y_temp[0], y_temp[1], y_temp[3]);
    k[stage][1] = dtheta_func(y_temp[0], y_temp[1], y_temp[4]);
    k[stage][2] = dphi_func(y_temp[0], y_temp[1], E, L);
    k[stage][3] = dp_r(y_temp[0], y_temp[1], y_temp[3], E, L, k_val);
    k[stage][4] = dp_theta(y_temp[0], y_temp[1], E, L);
  }
  // Combine the stages
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    y[i] += h / 6.0f * (k[0][i] + 2.0f * k[1][i] + 2.0f * k[2][i] + k[3][i]);
  }
}

// ---------------------------------------------------------------------
// Kernel: each thread simulates one ray.
// Input initial conditions are in the order:
// [r0, theta0, phi0, dr0, dtheta0, dphi0]
// The output trajectory (state vector per step) and the number of steps per ray
// are stored in contiguous device memory.
__global__ void simulateRayKernel(float3 pos, size_t num_rays_per_dim,
                                  float *trajectories, int *steps_out) {
  // printf("%.2f %.2f %.2f \n", pos.x ,pos.y, pos.z);
  int const idx = blockIdx.x * blockDim.x + threadIdx.x;
  int const num_rays = num_rays_per_dim * num_rays_per_dim;

  if (idx >= num_rays)
    return;

  int const idx_theta = idx / num_rays_per_dim;
  int const idx_phi = idx % num_rays_per_dim;

  float theta = (M_PI * idx_theta) / num_rays_per_dim;
  float phi = (2.0f * M_PI);
  // @TODO: (Investigate); Might need to rotate outgoing dirs to account for
  // camera orientation
  float3 camPos = make_float3(pos.x, pos.y, pos.z); // camera world pos
  float3 forward =
      normalizef3(make_float3(-camPos.x, // since modelCenter == (0,0,0)
                              -camPos.y, -camPos.z));

  float3 right = normalizef3(crossf3(forward, worldUp));
  float3 upVec = crossf3(right, forward);

  // now build your ray as before:
  float sinT = sinf(theta), cosT = cosf(theta);
  float sinP = sinf(phi), cosP = cosf(phi);

  float3 dir = sinT * (cosP * right + sinP * upVec) + cosT * forward;

  dir = normalizef3(dir);
  //   printf("%.2f %.2f %.2f \n", dir.x, dir.y, dir.z);

  float const x_vel = M_C * dir.x;
  float const y_vel = M_C * dir.y;
  float const z_vel = M_C * dir.z;

  float const A = c_a * c_rs / 2;

  float bl[6];
  cartesian_to_boyer_lindquist(pos.x, x_vel, pos.y, y_vel, pos.z, z_vel, A, bl);

  float const r0 = 2.0f / c_rs * bl[0];
  float const theta0 = bl[2];
  float const phi0 = bl[4];
  float const dr0 = bl[1] / M_C;
  float const dtheta0 = bl[3] * c_rs / (2.0f * M_C);
  float const dphi0 = bl[5] * c_rs / (2.0f * M_C);

  printf("%.2f %.2f %.2f %.2f %.2f %.2f \n", r0, theta0, phi0, dr0, dtheta0,
         dphi0);

  // Compute conserved quantities using Kerr equations.
  float E = E_func(r0, theta0, dr0, dtheta0, dphi0);
  float L = L_func(r0, theta0, dphi0, E);
  float k_val = k_func(r0, theta0, dr0, E, L);

  // Compute initial momenta.
  float S = sigma(r0, theta0);
  float p_r0 = S * dr0 / delta_r(r0);
  float p_theta0 = S * dtheta0;

  // Set up the initial state vector: [r, theta, phi, p_r, p_theta]
  float y[5];
  y[0] = r0;
  y[1] = theta0;
  y[2] = phi0;
  y[3] = p_r0;
  y[4] = p_theta0;
  // printf("%.2f, %.2f\n", theta0, phi0);

  // Pointer to this ray's trajectory data.
  float *ray_traj = &trajectories[idx * c_num_steps * 5];

  int step;
  for (step = 0; step < c_num_steps; step++) {
    // Terminate integration if ray is inside the horizon or outside the
    // environment.
    if (y[0] < 1.99 || y[0] > r_env_val)
      break;

    // Store the current state into the trajectory.
    for (int j = 0; j < 5; j++) {
      ray_traj[step * 5 + j] = y[j];
    }
    // Advance one RK4 step.
    rk4(y, c_h, E, L, k_val);
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
void simulateRays(float x, float y, float z, const float rs, const float Kerr,
    unsigned int num_rays_per_dim, unsigned int num_steps,
    float *trajectories_host, int *steps_out_host) {

  size_t num_rays = num_rays_per_dim * num_rays_per_dim;
  // Calculate sizes for memory allocation.
  size_t traj_size = num_rays * num_steps * 5 * sizeof(float);
  size_t steps_size = num_rays * sizeof(int);

  cudaMemcpyToSymbol(c_a, &Kerr, sizeof(float));
  cudaMemcpyToSymbol(c_rs, &rs, sizeof(float));
  cudaMemcpyToSymbol(c_num_steps, &num_steps, sizeof(unsigned int));

  // Allocate device memory.
  float *d_trajectories = nullptr;
  int *d_steps_out = nullptr;
  cudaMalloc(&d_trajectories, traj_size);
  cudaMalloc(&d_steps_out, steps_size);

  // Determine kernel launch configuration.
  int threadsPerBlock = 256;
  int blocks = (int)((num_rays + threadsPerBlock - 1) / threadsPerBlock);

  // Launch the simulation kernel.
  simulateRayKernel<<<blocks, threadsPerBlock>>>(
      make_float3(x, y, z), num_rays_per_dim, d_trajectories, d_steps_out);
  cudaDeviceSynchronize();

  // Copy the results back to host.
  cudaMemcpy(trajectories_host, d_trajectories, traj_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(steps_out_host, d_steps_out, steps_size, cudaMemcpyDeviceToHost);

  // Free device memory.
  cudaFree(d_trajectories);
  cudaFree(d_steps_out);
}

} // extern "C"