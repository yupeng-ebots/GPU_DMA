#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <cuda.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr size_t BLOCK_SIZE = 128*1024; // 128KB
// constexpr size_t BLOCK_SIZE = 10;
constexpr size_t BUFFER_SIZE = 4; // 4 times of BLOCK_SIZE
#define WRITE_FREQ 0
#define READ_FREQ 0
#define DATA_SIZE 1747*BLOCK_SIZE
// #define DATA_SIZE 7*BLOCK_SIZE

#define WIDTH 1944
#define HEIGHT 1472
#define NUM_FRAME 20


// std::atomic<size_t> write_ptr(0);

// std::atomic<size_t> read_ptr(0);

int write_ptr = 0;
int read_ptr = 0;

char* ring_buffer;
char* gpu_data_mem_head;
char* gpu_data_mem;
char* cpu_data_mem_head;
char* cpu_data_mem;

std::atomic <bool> producer_running(false);
std::atomic <bool> consumer_running(false);

__global__ void printAddress(char* variable) {
    printf("Variable address on GPU: %p\n", variable);
    for (int i = 0; i < 10; ++i) {
        printf("    Value at address %p: %d\n", variable + i, *(variable + i));
    }
}

void Producer() {
    while(producer_running) {
        // cudaMemcpy(&ring_buffer[write_ptr * BLOCK_SIZE], gpu_data_mem, BLOCK_SIZE, cudaMemcpyDeviceToHost);
        // make a void* pointer = &ring_buffer[write_ptr * BLOCK_SIZE]
        // float time;
        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);
        std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tp = std::chrono::high_resolution_clock::now();
        auto duration = tp.time_since_epoch();
        // std::cout << "P begin: " << duration.count() << "ns\n";
        cudaMemcpy(&ring_buffer[(write_ptr%BUFFER_SIZE) * BLOCK_SIZE], gpu_data_mem, BLOCK_SIZE, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        tp = std::chrono::high_resolution_clock::now();
        duration = tp.time_since_epoch();
        std::cout << "P end: " << duration.count() << "ns\n";
        
        // std::cout << "P end: " << std::chrono::system_clock::now() << "\n";
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // cudaEventElapsedTime(&time, start, stop);
        // std::cout << "Producer time: " << time << " ms\n";
        if (gpu_data_mem == gpu_data_mem_head + DATA_SIZE - BLOCK_SIZE) {
            producer_running = false;
        } else {
            gpu_data_mem += BLOCK_SIZE;
        }
        write_ptr++;
        // std::this_thread::sleep_for(std::chrono::milliseconds(WRITE_FREQ));
        std::this_thread::sleep_for(std::chrono::nanoseconds(WRITE_FREQ));
    }
}

void Consumer() {
    while(consumer_running) {
        if (read_ptr < write_ptr) {
            // memcpy(cpu_data_mem, &ring_buffer[read_ptr * BLOCK_SIZE], BLOCK_SIZE);
            // float time;
            // cudaEvent_t start, stop;
            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);
            // cudaEventRecord(start, 0);
            // std::cout << "C begin: " << std::chrono::system_clock::now() << "\n";
            std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> tp = std::chrono::high_resolution_clock::now();
            auto duration = tp.time_since_epoch();
            // std::cout << "C begin: " << duration.count() << "ns\n";
            cudaMemcpy(cpu_data_mem, &ring_buffer[(read_ptr%BUFFER_SIZE) * BLOCK_SIZE], BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            tp = std::chrono::high_resolution_clock::now();
            duration = tp.time_since_epoch();
            std::cout << "C end: " << duration.count() << "ns\n";
            // std::cout << "C begin: " << begin << " end: " << end << "\n\n";
            // std::cout << "C end: " << std::chrono::system_clock::now() << "\n";

            // cudaEventRecord(stop, 0);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&time, start, stop);
            // std::cout << "Consumer time: " << time << " ms\n";
            if (cpu_data_mem == cpu_data_mem_head + DATA_SIZE - BLOCK_SIZE) {
                consumer_running = false;
            } else {
                cpu_data_mem = cpu_data_mem + BLOCK_SIZE;
            }
            read_ptr++;
            // std::this_thread::sleep_for(std::chrono::milliseconds(READ_FREQ));
            std::this_thread::sleep_for(std::chrono::nanoseconds(READ_FREQ));
        }
    }
}

bool LoadImage(std::string image_folder, char* data_dst) {
    int width = WIDTH;
    int height = HEIGHT;
    int num_frame = NUM_FRAME;
    // int data_size = width * height * num_frame;
    // printf("data_size: %lu\n", data_size*sizeof(float));
    // cast data_dst to float
    float* data_dst_float = reinterpret_cast<float*>(data_dst);
    // load image
    for (int img_idx = 0; img_idx < num_frame; ++img_idx) {
        std::string img_path = image_folder + "/Frame" + std::to_string(img_idx) + ".tiff";
        cv::Mat cv_img1 = cv::imread(img_path.c_str(), cv::IMREAD_ANYDEPTH);
        cv::Mat cv_floatImg;
        cv_img1.convertTo(cv_floatImg, CV_32FC1);
        if (cv_img1.empty()) {
            std::cout << "Could not read the image: " << img_path << std::endl;
            return false;
        }
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {
                data_dst_float[img_idx * width * height + j * width + k] = cv_floatImg.at<float>(j, k)/4095.0f;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: ./ring_buffer_test <number_test> <image_folder>" << std::endl;
        return -1;
    }
    int number_test = atoi(argv[1]);
    std::string image_folder = argv[2];
    cudaMalloc((void**)&ring_buffer, BUFFER_SIZE * BLOCK_SIZE * sizeof(char));
    cudaMalloc((void**)&gpu_data_mem_head, DATA_SIZE * sizeof(char));
    for (int test_idx = 0; test_idx < number_test; ++test_idx) {
        // std::cout << "Test " << test_idx << std::endl;
        char* cpu_data = new char[DATA_SIZE];
        // ring_buffer = new char[BUFFER_SIZE * BLOCK_SIZE];
        
        std::cout << "Init Ring Buffer ";
        // printAddress<<<1, 1>>>(&ring_buffer[write_ptr * BLOCK_SIZE]);
        cpu_data_mem_head = new char[DATA_SIZE];

        for (int i = 0; i < DATA_SIZE; ++i) {
            cpu_data[i] = i%256;
        }

        char* test_image_date = new char[WIDTH*HEIGHT*NUM_FRAME*sizeof(float)];
        LoadImage(image_folder, cpu_data);


        
        // std::cout << "gpu_buffer size: " << DATA_SIZE * sizeof(char) << std::endl;

        cudaMemcpy(gpu_data_mem_head, cpu_data, DATA_SIZE, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpu_data_mem = gpu_data_mem_head;
        
        cpu_data_mem = cpu_data_mem_head;
        producer_running = true;
        consumer_running = true;
        std::thread producer(Producer);
        std::thread consumer(Consumer);

        consumer.join();
        producer.join();

        cudaDeviceSynchronize();
        // std::cout << "Data transfer is done" << std::endl;
        bool is_correct = true;
        for (int i = 0; i < DATA_SIZE; ++i) {
            if (cpu_data_mem_head[i] != cpu_data[i]) {
                // printf("Test: %d cpu_data_mem_head[%d]: %p  %d %d\n", test_idx, i, cpu_data_mem_head+i, *(cpu_data_mem_head+i), *(cpu_data+i));    
                is_correct = false;
            }
        }
        if (!is_correct) {
            std::cout << "Test " << test_idx << " Data transfer is wrong" << std::endl;
        } else {
            std::cout << "Test " << test_idx << " Data transfer is correct" << std::endl;
        }
        
        
        //convert cpu_data back to float and save as tiff image
        float* cpu_data_float = reinterpret_cast<float*>(cpu_data_mem_head);
        for (int img_idx = 0; img_idx < NUM_FRAME; ++img_idx) {
            cv::Mat cv_img1 = cv::Mat(HEIGHT, WIDTH, CV_32FC1, cpu_data_float + img_idx * WIDTH * HEIGHT);
            cv::Mat cv_img2;
            cv_img1.convertTo(cv_img2, CV_16UC1, 4095.0f);
            std::string img_path = image_folder + "/out_Frame" + std::to_string(img_idx) + ".tiff";
            cv::imwrite(img_path.c_str(), cv_img2);
        }

        delete[] cpu_data;
        delete[] cpu_data_mem_head;
        
    }
    cudaFree(ring_buffer);
    cudaFree(gpu_data_mem_head);
        
    

    return 0;    
}
