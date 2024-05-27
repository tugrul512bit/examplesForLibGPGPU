// compares 18000 words with each other and outputs a binary matrix where 1 = within 1 letter difference, 0 = different
// rtx4070: 8 milliseconds, including data-copy through pcie bridge (pcie v4.0 x16 bandwidth)

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"
int main()
{

    try
    {
        const int numWords = 18000;
        const int bufferSize = 1024*1024*32; 

        GPGPU::Computer computer(GPGPU::Computer::DEVICE_GPUS); 


        computer.compile(
            
            R"(

            kernel void findNeightbors( 
                global char * data,
                global int * start,
                global int * length,
                global char * matrix) 
            { 
                const int threadId=get_global_id(0); 
                const int wStart1 = start[threadId];
                const int wLength1 = length[threadId];
                const int wEnd1 = wStart1+wLength1;
                // assuming 20 letters are enough for longest word
                char localWord1[20];
                char localWord2[20];

                // load word1 into registers
                for(int i=wStart1;i<wEnd1;i++)
                {
                    localWord1[i-wStart1]=data[wStart1];
                }
    
                for(int j=0;j<18000;j++)
                {
                    const int wStart2 = start[j];
                    const int wLength2 = length[j];
                    const int wEnd2 = wStart2+wLength2;
                    // load word2 into local memory
                    for(int i=wStart2;i<wEnd2;i++)
                    {
                        localWord2[i-wStart2]=data[wStart2];
                    }

                    // compare
                    const int nLow = wLength1 < wLength2 ? wLength1 : wLength2;
                    int diff = abs(wLength1 - wLength2);
                    for(int i=0;i<nLow;i++)
                        diff += localWord1[i] != localWord2[i];
                    if(diff>1)
                    {
                        matrix[threadId + j*18000]=0;
                    }
                    else
                    {
                        matrix[threadId + j*18000]=1;
                    }
                }


             })", "findNeightbors");
                
        auto data = computer.createArrayInput<char>("data", bufferSize);        
        auto start = computer.createArrayInput<int>("start", numWords);
        auto length = computer.createArrayInput<int>("length", numWords);
        auto matrix = computer.createArrayOutput<char>("matrix", numWords*numWords);
        int currentIndex = 0;
        for (int i = 0; i < numWords; i++)
        {
            constexpr int currentWordSize = 4;
            char word [currentWordSize] = {'w','o','r','d'};
            start.access<int>(i) = currentIndex;
            length.access<int>(i) = currentWordSize;

            data.access<char>(currentIndex)=word[0];
            data.access<char>(currentIndex+1) = word[1];
            data.access<char>(currentIndex+2) = word[2];
            data.access<char>(currentIndex+3) = word[3];
            currentIndex += currentWordSize;
        }
        auto kernelParams = data.next(start).next(length).next(matrix);
        
        // benchmark for 20 times
        for (int i = 0; i < 20; i++)
        {
            size_t nanoSeconds;
            {
                GPGPU::Bench bench(&nanoSeconds);
                computer.compute(kernelParams, "findNeightbors", 0, numWords, 180 /* exact divider of numWords */);

                // word-0 vs word-1000 comparison result (same as word-1000 vs word-0)
                std::cout << (int)matrix.access<char>(1000) << std::endl;
            }
            std::cout << nanoSeconds / 1000000000.0f << " seconds" << std::endl; 
        }
        

    }
    catch (std::exception& ex)
    {
        std::cout << ex.what() << std::endl; // any error is handled here
    }
    return 0;
}

