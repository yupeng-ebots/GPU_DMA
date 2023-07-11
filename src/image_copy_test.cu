#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>



// #define SIZE (5.885*20*1024*1024)/sizeof(float)  // Approximately 110MB of floats
#define SIZE (5723136*20)/sizeof(float)  // Approximately 110MB of floats
#define BLOCK_SIZE 256
#define RUNS 10000

double calculateSD(std::vector<double> data) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double std_dev = std::sqrt(sq_sum / data.size() - mean * mean);

 

    return std_dev;
}

// Main program
int main() {
    float *hostArray, *devArray;
    size_t size = SIZE * sizeof(float);
    // Allocate Host memory
    hostArray = (float*)malloc(size);
    // Allocate Device memory
    cudaMalloc((void**)&devArray, size);
    // Initialize Host memory
    for (int i = 0; i < SIZE; ++i) hostArray[i] = (float)i;
    std::vector<double> timesTo;
    std::vector<double> timesFrom;
    for (int i = 0; i < RUNS; i++) {
        // Copy to the Device memory
        auto start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(devArray, hostArray, size, cudaMemcpyHostToDevice);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        timesTo.push_back(elapsed.count());
        // Copy to the Host memory
        start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(hostArray, devArray, size, cudaMemcpyDeviceToHost);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        timesFrom.push_back(elapsed.count());
    }
    double meanTo = std::accumulate(timesTo.begin(), timesTo.end(), 0.0) / timesTo.size();
    double maxTo = *std::max_element(timesTo.begin(), timesTo.end());
    double minTo = *std::min_element(timesTo.begin(), timesTo.end());
    double stdDevTo = calculateSD(timesTo);
    double meanFrom = std::accumulate(timesFrom.begin(), timesFrom.end(), 0.0) / timesFrom.size();
    double maxFrom = *std::max_element(timesFrom.begin(), timesFrom.end());
    double minFrom = *std::min_element(timesFrom.begin(), timesFrom.end());
    double stdDevFrom = calculateSD(timesFrom);
    printf("CPU to GPU transfer:\nMean: %f ms, Max: %f ms, Min: %f ms, Std Dev: %f ms\n", meanTo, maxTo, minTo, stdDevTo);
    printf("GPU to CPU transfer:\nMean: %f ms, Max: %f ms, Min: %f ms, Std Dev: %f ms\n", meanFrom, maxFrom, minFrom, stdDevFrom);
    // save timesTo in to a txt file split by ","
    FILE *fp;
    fp = fopen("../log/timesTo.txt", "w+");
    for (int i = 0; i < timesTo.size(); i++) {
        fprintf(fp, "%f", timesTo[i]);
        if (i != timesTo.size() - 1) {
            fprintf(fp, ",");
        }
    }
    fclose(fp);
    // save timesFrom in to a txt file split by ","
    fp = fopen("../log/timesFrom.txt", "w+");
    for (int i = 0; i < timesFrom.size(); i++) {
        fprintf(fp, "%f", timesFrom[i]);
        if (i != timesFrom.size() - 1) {
            fprintf(fp, ",");
        }
    }


    // Free the Device and Host memory
    cudaFree(devArray);
    free(hostArray);
    return 0;

}