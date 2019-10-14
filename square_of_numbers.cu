#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

__global__ void add(int N, double *a,double *b)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < N)
    {
        b[tid] = a[tid]*a[tid];
    }

}

int main(int argc, char *argv[])
{
    int N;  //Problem Size
    int T = 10, B = 1;            // threads per block/blocks per grid
    double *a,*b;
    double *dev_a, *dev_b;

    for(N=10000000;N<=100000000;N=N+10000000)
    {
        printf("N = %d\n",N);
        a = (double*)malloc(sizeof(double)*N);
        b = (double*)malloc(sizeof(double)*N);

        cudaMalloc((void**)&dev_a,N * sizeof(double));
        cudaMalloc((void**)&dev_b,N * sizeof(double));

        for(int i=0;i<N;i++)
        {
                // load arrays with some numbers
                a[i] = i;
        }

        cudaMemcpy(dev_a, a , N*sizeof(double),cudaMemcpyHostToDevice);

        T = 1024;
        B = ceil(double(N)/T);

        clock_t start_time = clock();
        add<<<B,T>>>(N,dev_a,dev_b);
        cudaDeviceSynchronize();
        clock_t end_time = clock();
        
        double parallel_time = (double(end_time-start_time)/CLOCKS_PER_SEC);
        cudaMemcpy(b,dev_b,N*sizeof(double),cudaMemcpyDeviceToHost);

        start_time = clock();
        int i;
        for(i=0;i<N;i++)
        {
            b[i] = a[i] * a[i];
        }
        end_time = clock();

        double serial_time  = (double(end_time-start_time)/CLOCKS_PER_SEC);

        double speedup = serial_time/parallel_time;
        printf("N=%d, parallel_time = %lf, serial_time = %lf, speedup = %lf\n",N,parallel_time,serial_time,speedup);
        //printf("%lf %lf\n",a[5],b[5]);
        free(a);
        free(b);

        cudaFree(dev_a); // clean up
        cudaFree(dev_b);
    }

        return 0;
}