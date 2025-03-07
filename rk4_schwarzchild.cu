#include <cuda_runtime.h>
#include <vector>
#include "device_launch_parameters.h"

__device__ void geodesic(double u, double dudphi, double& out_du_dphi, double& out_d2u_dphi2, double rs) {
    out_du_dphi = dudphi;
    out_d2u_dphi2 = -u * (1 - 3. / 2. * rs * u);
}

__device__ void rk4_step(double& u, double& dudphi, double& phi, double h, double rs) {
    double k1_u, k1_dudphi, k2_u, k2_dudphi, k3_u, k3_dudphi, k4_u, k4_dudphi;

    geodesic(u, dudphi, k1_u, k1_dudphi, rs);
    geodesic(u + 0.5 * k1_u * h, dudphi + 0.5 * k1_dudphi * h, k2_u, k2_dudphi, rs);
    geodesic(u + 0.5 * k2_u * h, dudphi + 0.5 * k2_dudphi * h, k3_u, k3_dudphi, rs);
    geodesic(u + k3_u * h, dudphi + k3_dudphi * h, k4_u, k4_dudphi, rs);

    phi += h;
    u = u + (k1_u + 2 * k2_u + 2 * k3_u + k4_u) * h / 6;
    dudphi = dudphi + (k1_dudphi + 2 * k2_dudphi + 2 * k3_dudphi + k4_dudphi) * h / 6;
}

__global__ void solveGeodesicKernel(double rs, double envmap_r, double u_0, double* dudphi_0_values, double h, int num_paths, int num_steps, double* u_values, double* dudphi_values, double* phi_values, double* angles_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    double u = u_0;
    double dudphi = dudphi_0_values[idx];
    double phi = 0.0;

    // Perform the first RK4 step
    rk4_step(u, dudphi, phi, h, rs);

    double r = 1.0 / u;
    double r_0 = 1.0 / u_0;
    double a = r * sin(phi);
    double b = r * cos(phi) - r_0;

    // Store starting angle
    angles_out[idx * 3] = atan2(a, b);

    u_values[idx * num_steps] = u;
    dudphi_values[idx * num_steps] = dudphi;
    phi_values[idx * num_steps ] = phi;
    
    auto out_of_bounds = [&u, &envmap_r]() -> bool {
        return (1.0 / u > envmap_r);
    };

    auto inside_singularity = [&u, &rs]() -> bool {
        return (1.0 / u < rs);
    };
    
    for (int step = 1; step < num_steps && !out_of_bounds() && !inside_singularity(); step++) {
        
        rk4_step(u, dudphi, phi, h, rs);
        
        if(out_of_bounds()) break;
            
        u_values[idx * num_steps + step] = u;
        dudphi_values[idx * num_steps + step] = dudphi;
        phi_values[idx * num_steps + step] = phi;
    }

    angles_out[idx * 3 + 1] = phi;
    angles_out[idx * 3 + 2] = u;
}

extern "C" {
    __declspec(dllexport) void cuda_test(
        double rs, double envmap_r, int num_paths, int num_steps, double u_0,
        double* dudphi_0_values, double h, double* u_out, double* phi_out, double* angle_out) {

        double* d_dudphi_0_values;
        double* d_u_values;
        double* d_dudphi_values;
        double* d_phi_values;
        double* d_angle_values;
            
        // Allocate device memory
        cudaMalloc(&d_dudphi_0_values, num_paths * sizeof(double));
        cudaMalloc(&d_u_values, num_paths * num_steps * sizeof(double));
        cudaMalloc(&d_dudphi_values, num_paths * num_steps * sizeof(double));
        cudaMalloc(&d_phi_values, num_paths * num_steps * sizeof(double));
        cudaMalloc(&d_angle_values, num_paths * 3 * sizeof(double));

        // Copy initial velocity values to device
        cudaMemcpy(d_dudphi_0_values, dudphi_0_values, num_paths * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsPerBlock = 256;
        int numBlocks = (num_paths + threadsPerBlock - 1) / threadsPerBlock;
        solveGeodesicKernel << <numBlocks, threadsPerBlock >> > (rs, envmap_r, u_0, d_dudphi_0_values, h, num_paths, num_steps, d_u_values, d_dudphi_values, d_phi_values, d_angle_values);

        cudaMemcpy(u_out, d_u_values, num_paths * num_steps * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(phi_out, d_phi_values, num_paths * num_steps * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(angle_out, d_angle_values, num_paths * 3 * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_dudphi_0_values);
        cudaFree(d_u_values);
        cudaFree(d_dudphi_values);
        cudaFree(d_phi_values);
        cudaFree(d_angle_values);
    }
}
