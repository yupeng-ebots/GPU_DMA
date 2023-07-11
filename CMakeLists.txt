cmake_minimum_required(VERSION 3.11.3)
project(MyProject)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add your CUDA file(s) to the project
set(CUDA_SOURCE_FILES ./src/producer_consumer.cu)

# Set CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")  # Adjust the architecture as needed

# Add an executable target
cuda_add_executable(PCIeTest ${CUDA_SOURCE_FILES})

# Link with OpenCV
target_link_libraries(PCIeTest ${OpenCV_LIBRARIES})