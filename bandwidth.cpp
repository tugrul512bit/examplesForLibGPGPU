// video memory bandwidth test
// rtx4070: 444 GB/s for read+write for 32-bit integers & 1M threads
// requires 4GB video memory

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"
int main()
{
    try
    {
        const size_t n = 1024*1024*512;

        // 0-index = first graphics card
        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS,0);
        computer.compile(
            "#define N_BUFFER "+std::to_string(n)+"ull" +

            R"(

            kernel void bandwidth( 
                global int * data1,
                global int * data2) 
            { 
                const int threadId=get_global_id(0); 

                for(int i=0;i<N_BUFFER;i+=1024*1024)
                    data2[threadId + i]=data1[threadId + i];

            }

        )", "bandwidth");

        // createArrayState does not make any pcie-transfer. its for keeping states within graphics card
        // createArrayInput: input data of kernel, copied from RAM to VRAM
        // createArrayOutput: output data of kernel, copied from VRAM to RAM
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
