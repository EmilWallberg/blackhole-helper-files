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

__device__ void rkf45_step(double& u, double& dudphi, double& phi, double& h, double rs, double tol) {
    double k1_u, k1_dudphi;
    double k2_u, k2_dudphi;
    double k3_u, k3_dudphi;
    double k4_u, k4_dudphi;
    double k5_u, k5_dudphi;
    double k6_u, k6_dudphi;

    double y, dy, z, dz;
    const double h_min = 1e-8;
    const double h_max = 1;
    const double safety_factor = 0.84;
    const double increase_safety_factor = 0.05;
    const int max_iterations = 1000;

    double e_u = 1e9, e_du = 1e9, e = 1e9;
    size_t counter = 0;

    while ((e > tol) && counter < max_iterations) {
        counter++;

        geodesic(u, dudphi, k1_u, k1_dudphi, rs);
        k1_u *= h; k1_dudphi *= h;

        geodesic(u + 0.25 * k1_u, dudphi + 0.25 * k1_dudphi, k2_u, k2_dudphi, rs);
        k2_u *= h; k2_dudphi *= h;

        geodesic(u + (3.0 / 32.0) * k1_u + (9.0 / 32.0) * k2_u,
                 dudphi + (3.0 / 32.0) * k1_dudphi + (9.0 / 32.0) * k2_dudphi, 
                 k3_u, k3_dudphi, rs);
        k3_u *= h; k3_dudphi *= h;

        geodesic(u + (1932.0 / 2197.0) * k1_u - (7200.0 / 2197.0) * k2_u + (7296.0 / 2197.0) * k3_u,
                 dudphi + (1932.0 / 2197.0) * k1_dudphi - (7200.0 / 2197.0) * k2_dudphi + (7296.0 / 2197.0) * k3_dudphi,
                 k4_u, k4_dudphi, rs);
        k4_u *= h; k4_dudphi *= h;

        geodesic(u + (439.0 / 216.0) * k1_u - 8.0 * k2_u + (3680.0 / 513.0) * k3_u - (845.0 / 4104.0) * k4_u,
                 dudphi + (439.0 / 216.0) * k1_dudphi - 8.0 * k2_dudphi + (3680.0 / 513.0) * k3_dudphi - (845.0 / 4104.0) * k4_dudphi,
                 k5_u, k5_dudphi, rs);
        k5_u *= h; k5_dudphi *= h;

        geodesic(u - (8.0 / 27.0) * k1_u + 2.0 * k2_u - (3544.0 / 2565.0) * k3_u + (1859.0 / 4104.0) * k4_u - (11.0 / 40.0) * k5_u,
                 dudphi - (8.0 / 27.0) * k1_dudphi + 2.0 * k2_dudphi - (3544.0 / 2565.0) * k3_dudphi + (1859.0 / 4104.0) * k4_dudphi - (11.0 / 40.0) * k5_dudphi,
                 k6_u, k6_dudphi, rs);
        k6_u *= h; k6_dudphi *= h;

        y = u + (25.0 / 216.0) * k1_u + (1408.0 / 2565.0) * k3_u + (2197.0 / 4104.0) * k4_u - (1.0 / 5.0) * k5_u;
        dy = dudphi + (25.0 / 216.0) * k1_dudphi + (1408.0 / 2565.0) * k3_dudphi + (2197.0 / 4104.0) * k4_dudphi - (1.0 / 5.0) * k5_dudphi;

        z = u + (16.0 / 135.0) * k1_u + (6656.0 / 12825.0) * k3_u + (28561.0 / 56430.0) * k4_u - (9.0 / 50.0) * k5_u + (2.0 / 55.0) * k6_u;
        dz = dudphi + (16.0 / 135.0) * k1_dudphi + (6656.0 / 12825.0) * k3_dudphi + (28561.0 / 56430.0) * k4_dudphi - (9.0 / 50.0) * k5_dudphi + (2.0 / 55.0) * k6_dudphi;

        e_u = fabs(z - y);
        e_du = fabs(dz - dy);
        e = sqrt(e_u * e_u + e_du * e_du);

        double scale_factor = 1.0;
        if (e > tol) {
            scale_factor = safety_factor * pow(tol / (e + 1e-12), 0.25);
        } else {
            scale_factor = increase_safety_factor * pow(tol / (e + 1e-12), 0.2);
        }

        scale_factor = fmax(0.5, fmin(2.0, scale_factor));
        
        h *= scale_factor;
        h = fmax(h_min, fmin(h, h_max));
    }

    u = z;
    dudphi = dz;
    phi += h;
}



__global__ void solveGeodesicKernel(double rs, double envmap_r, double u_0, double* dudphi_0_values, double h, double tol, int num_paths, int num_steps, double* u_values, double* dudphi_values, double* phi_values, double* angles_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) return;

    double u = u_0;
    double dudphi = dudphi_0_values[idx];
    double phi = 0.0;

    u_values[idx * num_steps] = u;
    dudphi_values[idx * num_steps] = dudphi;
    phi_values[idx * num_steps] = phi;

    // Perform the first RK4 step
    //rk4_step(u, dudphi, phi, h, rs);
    rkf45_step(u, dudphi, phi, h, rs, tol);
    
    double r = 1.0 / u;
    double r_0 = 1.0 / u_0;
    double a = r * sin(phi);
    double b = r * cos(phi) - r_0;
    
    // Store starting angle
    angles_out[idx * 3] = atan2(a, b);
    
    u_values[idx * num_steps + 1] = u;
    dudphi_values[idx * num_steps + 1] = dudphi;
    phi_values[idx * num_steps + 1] = phi;
    
    auto out_of_bounds = [&u, &envmap_r]() -> bool {
        return (1.0 / u > envmap_r);
    };
    
    auto inside_singularity = [&u, &rs]() -> bool {
        return (1.0 / u < rs);
    };
    
    for (int step = 2; step < num_steps && !out_of_bounds() && !inside_singularity(); step++) {
        
        //rk4_step(u, dudphi, phi, h, rs);
        rkf45_step(u, dudphi, phi, h, rs, tol);
        
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
        double* dudphi_0_values, double h, double tol, double* u_out, double* phi_out, double* angle_out) {

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
        solveGeodesicKernel << <numBlocks, threadsPerBlock >> > (rs, envmap_r, u_0, d_dudphi_0_values, h, tol, num_paths, num_steps, d_u_values, d_dudphi_values, d_phi_values, d_angle_values);

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
