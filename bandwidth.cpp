// video memory bandwidth test
// rtx4070: 430 GB/s for read+write of 32-bit integers
// requires 8GB video memory

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"
int main()
{
    try
    {
        const size_t n = 1024*1024*1024;

        // 0-index = first graphics card
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS,0);
        computer.compile(
            R"(
            kernel void bandwidth( 
                global int * data1,
                global int * data2) 
            { 
                const int threadId=get_global_id(0); 
                const int n = 1024*1024*1024;
                #pragma unroll 32
                for(int i=0;i<n;i+=1024*1024)
                    data2[threadId + i]=data1[threadId + i];

             }
        )", "bandwidth");

        auto data1 = computer.createArrayState<int>("data1", n);
        auto data2 = computer.createArrayState<int>("data2", n);
        auto kernelParams = data1.next(data2);

        // benchmark for 200 times
        for (int i = 0; i < 200; i++)
        {
            size_t nanoSeconds;
            {
                GPGPU::Bench bench(&nanoSeconds);
                computer.compute(kernelParams, "bandwidth", 0, 1024*1024 /* kernel threads */, 1024 /* block threads */);
            }
            auto readBW = (n * sizeof(int) / nanoSeconds);
            auto writeBW = readBW;
            auto totalBW = readBW+writeBW;
            std::cout << readBW << " GB/s read + " << writeBW << " GB/s write = "<< totalBW <<" GB/s total" << std::endl;
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; 
    }
    return 0;
}
