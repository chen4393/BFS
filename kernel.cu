/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[idx];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      const unsigned int wasVisited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!wasVisited) {
        const unsigned int globalTail = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[globalTail] = neighbor;
      }// if
    }// for
    idx += blockDim.x * gridDim.x;
  }// while
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  __shared__ unsigned int nextLevelNodes_s[BLOCK_SIZE];  //  block-level privatized queue
  __shared__ unsigned int numNextLevelNodes_s, ourNumNextLevelNodes;
  
  if (threadIdx.x == 0)  numNextLevelNodes_s = 0;
  __syncthreads();
  
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[idx];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      const unsigned int wasVisited = atomicExch(&(nodeVisited[neighbor]), 1);//  mark it as visited
      if (!wasVisited) {
        const unsigned int myTail = atomicAdd(&numNextLevelNodes_s, 1);
        if (myTail < BLOCK_SIZE) {//  if not full, add it to block-level queues
          nextLevelNodes_s[myTail] = neighbor;
        } else {//  if full, add it to global queue
          numNextLevelNodes_s = BLOCK_SIZE;
          const unsigned int myGlobalTail = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[myGlobalTail] = neighbor;
        }//  if..else
      }//  if
    }//  for
    idx += blockDim.x * gridDim.x;
  }// while
  __syncthreads();
  
  if (threadIdx.x == 0)
    ourNumNextLevelNodes = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);//  beginning index of reserved section
  __syncthreads();
  
  //  coalesced writes to global nextLevelNodes array
  for (unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += blockDim.x)
    nextLevelNodes[ourNumNextLevelNodes + i] = nextLevelNodes_s[i];//  copy vertices from block queue to global queue
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

































}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for(unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
      ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if(!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }

}

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

