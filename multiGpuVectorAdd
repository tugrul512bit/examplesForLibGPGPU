// add 1 to all elements of a vector of length 64
// multi-gpu capable by load-balancing (writing result is meant to be contiguous within each device, not interleaved)

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"


int main()
{
    try
    {

        const size_t n = 64;

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS);
        computer.compile(
            R"(

            kernel void vecAdd(global int * A, const global int * B) 
            { 
                const int threadId=get_global_id(0); 
                A[threadId] = B[threadId] + 1;
            }

            )", "vecAdd");

        auto data1 =  computer.createArrayOutput<int>("A", n);
        auto data2 =  computer.createArrayInputLoadBalanced<int>("B", n);
        auto kernelParams = data1.next(data2);
                
        for (int i = 0; i < 5; i++)
        {
            data2.access<int>(15)=i;
            computer.compute(kernelParams, "vecAdd", 0, 64 /* kernel threads */, 4 /* block threads */);                                
            std::cout << data1.access<int>(15)<<std::endl;
        }
    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; 
    }
    return 0;
}
