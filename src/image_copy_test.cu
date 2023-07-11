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