#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

void printDevProp(cudaDeviceProp devProp)
{
        printf("GPU card name -  %s\n",devProp.name);
        printf("GPU Computation Minor Capability - %d\n",devProp.minor);
        printf("GPU Computation Major Capability -  %d\n",devProp.major);
        printf("Maximum number of block dimensions -  %d %d %d\n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
        printf("Maximum number of grid dimensions - %d %d %d\n",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
        printf("Total GPU Memory global(bytes) - %zu\n",devProp.totalGlobalMem);
        printf("Total GPU Memory const(bytes) - %zu\n",devProp.totalConstMem);
        printf("Shared Memory available per block(bytes) - %zu\n",devProp.sharedMemPerBlock);
        printf("Warp size (number of threads per warp) - %d\n",devProp.warpSize);

        printf("Clock frequency in kilohertz - %d\n",devProp.clockRate);
        printf("Number of multiprocessors on device - %d\n",devProp.multiProcessorCount);
        printf("32-bit registers available per block - %d\n",devProp.regsPerBlock);
        printf("Maximum number of threads per block - %d\n",devProp.maxThreadsPerBlock);
        printf("Device can concurrently copy memory and execute a kernel - %d\n",devProp.deviceOverlap);
        printf("Whether there is a run time limit on kernels - %d\n",devProp.kernelExecTimeoutEnabled);
        printf("Device is integrated as opposed to discrete - %d\n",devProp.integrated);
}
int main()
{
        int i;
        int devCount;
        cudaGetDeviceCount(&devCount);
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n", devCount);
        for (i = 0; i < devCount; ++i)
        {
                // Get device properties
                printf("\nCUDA Device #%d\n", i);
                cudaDeviceProp devProp;
                cudaGetDeviceProperties(&devProp, i);
                printDevProp(devProp);
        }
        return 0;
}