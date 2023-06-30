#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda.h>

constexpr size_t BLOCK_SIZE = 128*1024; // 128KB
constexpr size_t BUFFER_SIZE = 4; // 4 times of BLOCK_SIZE
#define WRITE_FREQ 20
#define READ_FREQ 1
#define DATA_SIZE 874*BLOCK_SIZE

std::atomic<size_t> write_ptr(0);
std::atomic<size_t> read_ptr(0);

char* ring_buffer = new char[BUFFER_SIZE * BLOCK_SIZE];
char* gpu_data_mem_head;
char* gpu_data_mem;
char* cpu_data_mem_head = new char[DATA_SIZE];
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
            std::this_thread::sleep_for(std::chrono::milliseconds(WRITE_FREQ));
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
            std::this_thread::sleep_for(std::chrono::milliseconds(READ_FREQ));
        }
    }
}

int main() {
    char* cpu_data = new char[DATA_SIZE];
    for (int i = 0; i < DATA_SIZE; ++i) {
        cpu_data[i] = i%256;
    }

    cudaMalloc((void**)&gpu_data_mem_head, DATA_SIZE * sizeof(char));
    std::cout << "gpu_buffer size: " << DATA_SIZE * sizeof(char) << std::endl;

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
    std::cout << "Data transfer is done" << std::endl;

    for (int i = 0; i < DATA_SIZE; ++i) {
        if (cpu_data_mem_head[i] != cpu_data[i]) {
            printf("cpu_data_mem_head[%d]: %p  %d %d\n", i, cpu_data_mem_head+i, *(cpu_data_mem_head+i), *(cpu_data+i));    
        }   
    }

    // Deallocate memory
    delete[] cpu_data;
    delete[] cpu_data_mem_head;
    delete[] ring_buffer;
    cudaFree(gpu_data_mem_head);

    return 0;
}
