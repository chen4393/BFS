/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    // Variables
    unsigned int numNodes;
    unsigned int maxNeighborsPerNode;
    unsigned int *nodePtrs_h;
    unsigned int *nodeNeighbors_h;
    unsigned int *nodeVisited_h;
    unsigned int *nodeVisited_ref; // Needed for reference checking
    unsigned int *currLevelNodes_h;
    unsigned int *nextLevelNodes_h;
    unsigned int *numCurrLevelNodes_h;
    unsigned int *numNextLevelNodes_h;
    unsigned int *nodePtrs_d;
    unsigned int *nodeNeighbors_d;
    unsigned int *nodeVisited_d;
    unsigned int *currLevelNodes_d;
    unsigned int *nextLevelNodes_d;
    unsigned int *numCurrLevelNodes_d;
    unsigned int *numNextLevelNodes_d;
    cudaError_t cuda_ret;

    enum Mode {CPU = 1, GPU_GLOBAL_QUEUE, GPU_BLOCK_QUEUE, GPU_WARP_QUEUE};
    Mode mode;

    if(argc == 2) {
        mode = (Mode) atoi(argv[1]);
        numNodes = 200000;
        maxNeighborsPerNode = 10;
    } else if(argc == 3) {
        mode = (Mode) atoi(argv[1]);
        numNodes = atoi(argv[2]);
        maxNeighborsPerNode = 10;
    } else if(argc == 4) {
        mode = (Mode) atoi(argv[1]);
        numNodes = atoi(argv[2]);
        maxNeighborsPerNode = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters."
        "\n"
        "\n    Usage: ./bfs-queuing <m>          # Mode: m, Nodes: 200,000, "
                                                    "Max neighbors per node: 10"
        "\n           ./bfs-queuing <m> <N>      # Mode: m, Nodes:       N, "
                                                    "Max neighbors per node: 10"
        "\n           ./bfs-queuing <m> <N> <M>  # Mode: m, Nodes:       N, "
                                                    "Max neighbors per node:  M"
        "\n"
        "\n    Modes: 1 = CPU"
        "\n           2 = GPU with global queuing"
        "\n           3 = GPU with block and global queuing"
        "\n           4 = GPU with warp, block, and global queuing"
        "\n\n");
        exit(0);
    }

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    setupProblem(numNodes, maxNeighborsPerNode, &nodePtrs_h, &nodeNeighbors_h,
      &nodeVisited_h, &nodeVisited_ref, &currLevelNodes_h, &nextLevelNodes_h,
      &numCurrLevelNodes_h, &numNextLevelNodes_h);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    # Nodes = %u\n    Max neighbors per node = %u\n",
      numNodes, maxNeighborsPerNode);

    // Allocate device variables ----------------------------------------------

    if(mode != CPU) {
        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMalloc((void**)&nodePtrs_d,
          (numNodes + 1)*sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&nodeVisited_d, numNodes*sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&nodeNeighbors_d,
          nodePtrs_h[numNodes]*sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&numCurrLevelNodes_d,
          sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&currLevelNodes_d,
          (*numCurrLevelNodes_h)*sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&numNextLevelNodes_d,
          sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cuda_ret = cudaMalloc((void**)&nextLevelNodes_d,
          (numNodes)*sizeof(unsigned int));
        if(cuda_ret!=cudaSuccess) FATAL("Unable to allocate device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    // Copy host variables to device ------------------------------------------

    if(mode != CPU) {
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(nodePtrs_d, nodePtrs_h,
          (numNodes + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cuda_ret = cudaMemcpy(nodeVisited_d, nodeVisited_h,
          numNodes*sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cuda_ret = cudaMemcpy(nodeNeighbors_d, nodeNeighbors_h,
          nodePtrs_h[numNodes]*sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cuda_ret = cudaMemcpy(numCurrLevelNodes_d, numCurrLevelNodes_h,
          sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cuda_ret = cudaMemcpy(currLevelNodes_d, currLevelNodes_h,
          (*numCurrLevelNodes_h)*sizeof(unsigned int), cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cuda_ret = cudaMemset(numNextLevelNodes_d, 0, sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) {
          FATAL("Unable to copy memory to the device");
        }

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel ");

    if(mode == CPU) {
        printf("(CPU)...");fflush(stdout);
        startTime(&timer);
        cpu_queuing(nodePtrs_h, nodeNeighbors_h, nodeVisited_h,
          currLevelNodes_h, nextLevelNodes_h, numCurrLevelNodes_h,
          numNextLevelNodes_h);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_GLOBAL_QUEUE) {
        printf("(GPU with global queuing)...");fflush(stdout);
        startTime(&timer);
        gpu_global_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
          currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
          numNextLevelNodes_d);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_BLOCK_QUEUE) {
        printf("(GPU with block and global queuing)...");fflush(stdout);
        startTime(&timer);
        gpu_block_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
          currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
          numNextLevelNodes_d);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_WARP_QUEUE) {
        printf("(GPU with warp, block, and global queuing)...");fflush(stdout);
        startTime(&timer);
        gpu_warp_queuing(nodePtrs_d, nodeNeighbors_d, nodeVisited_d,
          currLevelNodes_d, nextLevelNodes_d, numCurrLevelNodes_d,
          numNextLevelNodes_d);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else {
        printf("Invalid mode!\n");
        exit(0);
    }


    // Copy device variables from host ----------------------------------------

    if(mode != CPU) {

        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(numNextLevelNodes_h, numNextLevelNodes_d,
          sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

        cuda_ret = cudaMemcpy(nextLevelNodes_h, nextLevelNodes_d,
          numNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

        cuda_ret = cudaMemcpy(nodeVisited_h, nodeVisited_d,
          numNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(numNodes, nodePtrs_h, nodeNeighbors_h, nodeVisited_h,
      nodeVisited_ref, currLevelNodes_h, nextLevelNodes_h, numCurrLevelNodes_h,
      numNextLevelNodes_h);

    // Free memory ------------------------------------------------------------

    free(nodePtrs_h);
    free(nodeVisited_h);
    free(nodeVisited_ref);
    free(nodeNeighbors_h);
    free(numCurrLevelNodes_h);
    free(currLevelNodes_h);
    free(numNextLevelNodes_h);
    free(nextLevelNodes_h);
    if(mode != CPU) {
      cudaFree(nodePtrs_d);
      cudaFree(nodeVisited_d);
      cudaFree(nodeNeighbors_d);
      cudaFree(numCurrLevelNodes_d);
      cudaFree(currLevelNodes_d);
      cudaFree(numNextLevelNodes_d);
      cudaFree(nextLevelNodes_d);

    }

    return 0;
}

