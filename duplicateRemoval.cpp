/*
 brute-force for 100k elements:
 number of unique elements = 62900
percentage of unique elements = 62.9%
-------------------------------------------------
cpu duplicate removal =0.0138678s
number of uniques after duplicate removal = 62900
-------------------------------------------------
cpu duplicate removal (optimized) =0.0103347s
number of uniques after duplicate removal = 62900
-------------------------------------------------
cpu duplicate removal (optimized+) =0.0043014s
number of uniques after duplicate removal = 62900
-------------------------------------------------
cpu duplicate removal multithreaded =0.0182354s
number of uniques after duplicate removal = 62900
-------------------------------------------------
gpu duplicate removal brute-force O(N^2)=0.0075971s
number of uniques after duplicate removal = 62900
-------------------------------------------------
*/

#include <iostream>
#include <fstream>

#include "gpgpu.hpp"

#include<random>
#include<map>
#include<thread>

std::vector<int> GenerateDuplicates(const int n=1000000, const int lowerBound = 0, const int higherBound = 10000000)
{
    std::random_device rd; // random device engine, usually based on /dev/random on UNIX-like systems
    // initialize Mersennes' twister using rd to generate the seed
    std::mt19937 rng{ rd() };
    std::uniform_int_distribution<int> uid(lowerBound, higherBound);

    std::vector<int> result(n);
    for (int i = 0; i < n; i++)
    {
        result[i]=uid(rng);
    }
    return result;
}

int NumberOfUniqueElements(std::vector<int> dup)
{
    const int n = dup.size();
    std::map<int, int> numDuplicatesPerElement;
    for (int i = 0; i < n; i++)
    {
        numDuplicatesPerElement[dup[i]]++;
    }
    int result = 0;
    for (auto& e : numDuplicatesPerElement)
    {
        result++;
    }
    return result;
}

// cpu single-threaded duplicate removal
std::vector<int> RemoveDuplicatesCpu(std::vector<int> dup)
{
    const int n = dup.size();
    std::vector<int> result;
    std::map<int, int> numDuplicatesPerElement;
    for (int i = 0; i < n; i++)
    {
        numDuplicatesPerElement[dup[i]]++;
    }
    for (auto& e : numDuplicatesPerElement)
    {
        result.push_back(e.first);
    }
    return result;
}


// cpu single-threaded duplicate removal, optimized by sorting
std::vector<int> RemoveDuplicatesCpu2(std::vector<int> dup)
{
    std::sort(dup.begin(), dup.end());
    const int n = dup.size();
    std::vector<int> result;
    std::map<int, int> numDuplicatesPerElement;
    for (int i = 0; i < n; i++)
    {
        numDuplicatesPerElement[dup[i]]++;
    }
    for (auto& e : numDuplicatesPerElement)
    {
        result.push_back(e.first);
    }
    return result;
}

// cpu single-threaded duplicate removal, optimized by sorting + without map
std::vector<int> RemoveDuplicatesCpu3(std::vector<int> dup)
{    
    std::sort(dup.begin(), dup.end());
    const int n = dup.size();
    std::vector<int> result;
    int current = dup[0];
    for (int i = 1; i < n; i++)
    {
        const int element = dup[i];
        if (current != element)
        {
           result.push_back(current);
           current = element;
        }
    }
    
    result.push_back(current);
    return result;
}

// cpu multi-threaded duplicate removal
std::vector<int> RemoveDuplicatesCpuMulti(std::vector<int> dup)
{
    const int n = dup.size();
    std::vector<int> result;
    const int thr = 10;// threads
    std::vector<std::map<int, int>> numDuplicatesPerElement(thr);
    std::vector<std::thread> threads;
    for (int t = 0; t < thr; t++)
    {
        threads.emplace_back([&,t]() {
            for (int i = t * (n/thr); i < (t+1)*(n/thr); i++)
            {                
                if(i<n)
                    numDuplicatesPerElement[t][dup[i]]++;
            }

            if(t == thr-1)
            if ((t + 1) * (n / thr) < n - 1)
            {
                for(int i= (t + 1) * (n / thr);i<n;i++)
                    numDuplicatesPerElement[t][dup[i]]++;
            }
        });

    }
    for (int i = 0; i < thr; i++)
        threads[i].join();

    for (int i = 1; i < thr; i++)
    {
        for (auto& e : numDuplicatesPerElement[i])
        {
            numDuplicatesPerElement[0][e.first] += e.second;
        }
    }

    for (auto& e : numDuplicatesPerElement[0])
    {
        result.push_back(e.first);
    }
    return result;
}
// gpu-accelerated duplicate removal
struct GpuDuplicateRemover
{
    GpuDuplicateRemover(const int nElements=1000000):computer(GPGPU::Computer::DEVICE_GPUS, 0/*select only first gpu*/),n(nElements)
    {
        try
        {
            computer.compile(
                R"(

            kernel void findDuplicate(const global int * input, global int * output, const int numElements) 
            { 
                const int threadId=get_global_id(0); 
                if(threadId >= numElements) return;
                const int val = input[threadId];
                int repeats = 0;
                int firstIndex = -1;
                for(int i=0;i<numElements;i++)
                {                   
                    if(val == input[i])
                    {
                        if(firstIndex == -1)
                            firstIndex = i;
                        repeats++;
                    }
                }
                if(threadId == firstIndex)
                {
                    output[threadId]=val;
                }
                else
                {
                    output[threadId]=-repeats;
                }
            }

            )", "findDuplicate");

            input = computer.createArrayInput<int>("input", n);
            output = computer.createArrayOutputAll<int>("output", n);
            numElements = computer.createScalarInput<int>("numElements");
            numElements = n;
            kernelParams = input.next(output).next(numElements);
        }
        catch (std::exception& ex)
        {
            std::cout << ex.what() << std::endl;
        }
    }

    GPGPU::Computer computer;
    GPGPU::HostParameter input, output, numElements, kernelParams;
    int n;
    std::vector<int> RemoveDuplicatesGpuBruteForce(std::vector<int> dup)
    {
        try
        {
            input.copyDataFromPtr(dup.data());
            computer.compute(kernelParams, "findDuplicate", 0, n + (256 - (n % 256)) /* kernel threads */, 256 /* block threads */);
            output.copyDataToPtr(dup.data());
        }
        catch (std::exception& ex)
        {
            std::cout << ex.what() << std::endl;
        }

        std::vector<int> result;
        for (int i = 0; i < n; i++)
            if (dup[i] >= 0)
            {
                result.push_back(dup[i]);
            }
        return result;
    }
};

int main()
{
    const int n = 100000;
    std::cout << "n=" << n << std::endl;
    std::vector<int> duplicates;
    size_t t;
    duplicates = GenerateDuplicates(n,0,n);


    int numUnique = NumberOfUniqueElements(duplicates);
    std::cout << "number of unique elements = " << numUnique << std::endl;
    std::cout << "percentage of unique elements = " << 100.0f * numUnique / (float)n << "%" << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    std::vector<int> cpuUnduplicated;
    // warming up
    for (int i = 0; i < 10; i++)
        cpuUnduplicated = RemoveDuplicatesCpu(duplicates);
    {
        GPGPU::Bench bench(&t);
        cpuUnduplicated = RemoveDuplicatesCpu(duplicates);
    }
    std::cout << "cpu duplicate removal =" << t / 1000000000.0f << "s" << std::endl;
    int numUndup = cpuUnduplicated.size();
    std::cout << "number of uniques after duplicate removal = " << numUndup << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    std::vector<int> cpuUnduplicated2;
    // warming up
    for (int i = 0; i < 10; i++)
        cpuUnduplicated2 = RemoveDuplicatesCpu2(duplicates);
    {
        GPGPU::Bench bench(&t);
        cpuUnduplicated2 = RemoveDuplicatesCpu2(duplicates);
    }
    std::cout << "cpu duplicate removal (optimized) =" << t / 1000000000.0f << "s" << std::endl;
    int numUndup2 = cpuUnduplicated2.size();
    std::cout << "number of uniques after duplicate removal = " << numUndup2 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    std::vector<int> cpuUnduplicated3;
    // warming up
    for (int i = 0; i < 10; i++)
        cpuUnduplicated3 = RemoveDuplicatesCpu3(duplicates);
    {
        GPGPU::Bench bench(&t);
        cpuUnduplicated3 = RemoveDuplicatesCpu3(duplicates);
    }
    std::cout << "cpu duplicate removal (optimized+) =" << t / 1000000000.0f << "s" << std::endl;
    int numUndup3 = cpuUnduplicated3.size();
    std::cout << "number of uniques after duplicate removal = " << numUndup3 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    std::vector<int> cpuUnduplicated4;
    // warming up
    for (int i = 0; i < 10; i++)
        cpuUnduplicated4 = RemoveDuplicatesCpuMulti(duplicates);
    {
        GPGPU::Bench bench(&t);
        cpuUnduplicated4 = RemoveDuplicatesCpuMulti(duplicates);
    }
    std::cout << "cpu duplicate removal multithreaded =" << t / 1000000000.0f << "s" << std::endl;
    int numUndup4 = cpuUnduplicated4.size();
    std::cout << "number of uniques after duplicate removal = " << numUndup4 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;

    GpuDuplicateRemover gpu(n);
    std::vector<int> gpuUnduplicated;
    // warming up
    for(int i=0;i<10;i++)
        gpuUnduplicated = gpu.RemoveDuplicatesGpuBruteForce(duplicates);
    {
        GPGPU::Bench bench(&t);
        gpuUnduplicated = gpu.RemoveDuplicatesGpuBruteForce(duplicates);
    }
    std::cout << "gpu duplicate removal brute-force O(N^2)=" << t / 1000000000.0f << "s" << std::endl;
    int numUndup5 = gpuUnduplicated.size();
    std::cout << "number of uniques after duplicate removal = " << numUndup5 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
    return 0;

}
