#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

constexpr size_t BLOCK_SIZE = 128*1024; // 128KB
constexpr size_t BUFFER_SIZE = 4; // 4 times of BLOCK_SIZE
#define WRITE_FREQ 100
#define READ_FREQ 0
#define DATA_SIZE 1747*BLOCK_SIZE

#define WIDTH 1944
#define HEIGHT 1472
#define NUM_FRAME 20


std::atomic<size_t> write_ptr(0);
std::atomic<size_t> read_ptr(0);

char* ring_buffer;
char* gpu_data_mem_head;
char* gpu_data_mem;
char* cpu_data_mem_head;
char* cpu_data_mem;

std::atomic <bool> producer_running(false);
std::atomic <bool> consumer_running(false);

void Producer() {
    while(producer_running) {
        if ((write_ptr + 1) % BUFFER_SIZE != read_ptr.load(std::memory_order_acquire)) {
            cudaMemcpy(&ring_buffer[write_ptr * BLOCK_SIZE], gpu_data_mem, BLOCK_SIZE, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            if (gpu_data_mem == gpu_data_mem_head + DATA_SIZE - BLOCK_SIZE) {
                producer_running = false;
            } else {
                gpu_data_mem += BLOCK_SIZE;
            }
            write_ptr.store((write_ptr + 1) % BUFFER_SIZE, std::memory_order_release);
            // std::this_thread::sleep_for(std::chrono::milliseconds(WRITE_FREQ));
            std::this_thread::sleep_for(std::chrono::microseconds(WRITE_FREQ));
        }
    }
}

void Consumer() {
    while(consumer_running) {
        if (read_ptr.load(std::memory_order_acquire) != write_ptr) {
            memcpy(cpu_data_mem, &ring_buffer[read_ptr * BLOCK_SIZE], BLOCK_SIZE);
            if (cpu_data_mem == cpu_data_mem_head + DATA_SIZE - BLOCK_SIZE) {
                consumer_running = false;
            } else {
                cpu_data_mem = cpu_data_mem + BLOCK_SIZE;
            }
            read_ptr.store((read_ptr + 1) % BUFFER_SIZE, std::memory_order_release);
            // std::this_thread::sleep_for(std::chrono::milliseconds(READ_FREQ));
            std::this_thread::sleep_for(std::chrono::microseconds(READ_FREQ));
        }
    }
}

bool LoadImage(std::string image_folder, char* data_dst) {
    int width = WIDTH;
    int height = HEIGHT;
    int num_frame = NUM_FRAME;
    int data_size = width * height * num_frame;
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

    for (int test_idx = 0; test_idx < number_test; ++test_idx) {
        // std::cout << "Test " << test_idx << std::endl;
        char* cpu_data = new char[DATA_SIZE];
        ring_buffer = new char[BUFFER_SIZE * BLOCK_SIZE];
        cpu_data_mem_head = new char[DATA_SIZE];

        for (int i = 0; i < DATA_SIZE; ++i) {
            cpu_data[i] = i%256;
        }

        char* test_image_date = new char[WIDTH*HEIGHT*NUM_FRAME*sizeof(float)];
        LoadImage(image_folder, cpu_data);


        cudaMalloc((void**)&gpu_data_mem_head, DATA_SIZE * sizeof(char));
        // std::cout << "gpu_buffer size: " << DATA_SIZE * sizeof(char) << std::endl;

        cudaMemcpy(gpu_data_mem_head, cpu_data, DATA_SIZE, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        gpu_data_mem = gpu_data_mem_head;
        
        cpu_data_mem = cpu_data_mem_head;
        producer_running = true;
        consumer_running = true;
        std::thread producer(Producer);
        std::thread consumer(Consumer);

        producer.join();
        consumer.join();

        cudaDeviceSynchronize();
        // std::cout << "Data transfer is done" << std::endl;

        for (int i = 0; i < DATA_SIZE; ++i) {
            if (cpu_data_mem_head[i] != cpu_data[i]) {
                printf("Test: %d cpu_data_mem_head[%d]: %p  %d %d\n", test_idx, i, cpu_data_mem_head+i, *(cpu_data_mem_head+i), *(cpu_data+i));    
                // break;
            }   
        }
        std::cout << "Test " << test_idx << " Data transfer is correct" << std::endl;
        
        //convert cpu_data back to float and save as tiff image
        float* cpu_data_float = reinterpret_cast<float*>(cpu_data);
        for (int img_idx = 0; img_idx < NUM_FRAME; ++img_idx) {
            cv::Mat cv_img1 = cv::Mat(HEIGHT, WIDTH, CV_32FC1, cpu_data_float + img_idx * WIDTH * HEIGHT);
            cv::Mat cv_img2;
            cv_img1.convertTo(cv_img2, CV_16UC1, 4095.0f);
            std::string img_path = image_folder + "/Frame" + std::to_string(img_idx) + "_out.tiff";
            cv::imwrite(img_path.c_str(), cv_img2);
        }

        // Deallocate memory
        delete[] cpu_data;
        delete[] cpu_data_mem_head;
        delete[] ring_buffer;
        cudaFree(gpu_data_mem_head);
    }
        
    

    return 0;    
}
