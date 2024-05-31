// add 1 to all elements of a vector of length 64
// single-gpu version

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"


int main()
{
    try
    {

        const size_t n = 64;

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS,0/*select only first gpu*/);
        computer.compile(
            R"(

            kernel void vecAdd(global int * A, const global int * B) 
            { 
                const int threadId=get_global_id(0); 
                A[threadId] = B[threadId] + 1;
            }

            )", "vecAdd");

        auto data1 =  computer.createArrayOutputAll<int>("A", n); // writes all results and assumes single gpu is used
        auto data2 =  computer.createArrayInput<int>("B", n); // loads all elements (broadcasts to all selected gpus)
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
