# GPU_DMA
Pseudo DMA to transfer data from GPU to CPU
## command line on server
nvcc ./src/producer_consumer.cu -o ./bin/pc `pkg-config --cflags --libs opencv`; ./bin/pc 10 ./images/