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
#define NUM_WQUEUES_PER_WARP 1
#define NUM_WQUEUES (NUM_WARPS*NUM_WQUEUES_PER_WARP)

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
<<<<<<< HEAD
    unsigned int node = currLevelNodes[idx];
=======
    const unsigned int node = currLevelNodes[idx];
>>>>>>> ad66ce81e6162113499c4ca15021e65301634553
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      const unsigned int wasVisited = atomicExch(&(nodeVisited[neighbor]), 1);
      if (!wasVisited) {
        const unsigned int globalTail = atomicAdd(numNextLevelNodes, 1);
<<<<<<< HEAD
	nextLevelNodes[globalTail] = neighbor;
=======
        nextLevelNodes[globalTail] = neighbor;
>>>>>>> ad66ce81e6162113499c4ca15021e65301634553
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
  __shared__ unsigned int nextLevelNodes_s[BQ_CAPACITY];  //  block-level privatized queue
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
        if (myTail < BQ_CAPACITY) {//  if not full, add it to block-level queues
          nextLevelNodes_s[myTail] = neighbor;
        } else {//  if full, add it to global queue
          numNextLevelNodes_s = BQ_CAPACITY;
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
  // block-level privatized queue
  __shared__ unsigned int nextLevelNodes_s[BQ_CAPACITY];  
  __shared__ unsigned int numNextLevelNodes_s, ourNumNextLevelNodes;

  // warp-level queues
  __shared__ unsigned int wQueues[WQ_CAPACITY][NUM_WQUEUES];
  __shared__ unsigned int numWqTails[NUM_WQUEUES], ourBqSP[NUM_WQUEUES];
  
  // warp ID and warp offset
  const unsigned int warpID = threadIdx.x % NUM_WQUEUES;
  const unsigned int warpOffset = threadIdx.x / NUM_WQUEUES;

  if (threadIdx.x == 0) {
    numNextLevelNodes_s = 0;
    for (int i = 0; i < NUM_WQUEUES; i++)
      numWqTails[i] = 0;
  }
  __syncthreads();
  
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  while (idx < *numCurrLevelNodes) {
    const unsigned int node = currLevelNodes[idx];
    for (unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      const unsigned int wasVisited = atomicExch(&(nodeVisited[neighbor]), 1);//  mark it as visited
      
      if (!wasVisited) {
	const unsigned int myWqTail = atomicAdd(&(numWqTails[warpID]), 1);

//wQueues[myWqTail][warpID] = neighbor;

	if (myWqTail < WQ_CAPACITY) {//	if not full, add it to warp-level queues
	  wQueues[myWqTail][warpID] = neighbor;
	} else {//  same as block + global version
	    numWqTails[warpID] = WQ_CAPACITY;
	    const unsigned int myBqTail = atomicAdd(&numNextLevelNodes_s, 1);
            if (myBqTail < BQ_CAPACITY) {//  if not full, add it to block-level queues
              nextLevelNodes_s[myBqTail] = neighbor;
            } else {//  if full, add it to global queue
              numNextLevelNodes_s = BQ_CAPACITY;
              const unsigned int myGqTail = atomicAdd(numNextLevelNodes, 1);
              nextLevelNodes[myGqTail] = neighbor;
            }//  if..else
	}//  if..else

      }//  if
    }//  for
    idx += blockDim.x * gridDim.x;
  }// while
  __syncthreads();

  // Determine the starting point of copying contents from warp-level queues to block-level queue
  if (warpOffset == 0) {
    //  beginning index of reserved section of block queue
    ourBqSP[warpID] = atomicAdd(&numNextLevelNodes_s, numWqTails[warpID]);
    //  beginning index of reserved section of global queue
    //ourNumNextLevelNodes = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);
  }
  __syncthreads();

  //  coalesced writes to block-level queue from warp-level queues
  for (unsigned int i = warpOffset; i < numWqTails[warpID]; i += WARP_SIZE) {
      const unsigned int myBqTail = numWqTails[warpID] + ourBqSP[warpID];
      //nextLevelNodes_s[ourBqSP[warpID] + i] = wQueues[i][warpID];

      if (myBqTail < BQ_CAPACITY) {
        // copy vertices from warp queue to block queue
        nextLevelNodes_s[ourBqSP[warpID] + i] = wQueues[i][warpID];
      } else {
	numNextLevelNodes_s = BQ_CAPACITY;
        // copy to global queue
	nextLevelNodes[ourNumNextLevelNodes + i] = wQueues[i][warpID];
      }

  }
  __syncthreads();

  if (threadIdx.x == 0) {
    //  beginning index of reserved section of global queue
    ourNumNextLevelNodes = atomicAdd(numNextLevelNodes, numNextLevelNodes_s);
  }
  __syncthreads();

  //  coalesced writes to global nextLevelNodes array
  for (unsigned int i = threadIdx.x; i < numNextLevelNodes_s; i += blockDim.x) {
    //  copy vertices from block queue to global queue
    nextLevelNodes[ourNumNextLevelNodes + i] = nextLevelNodes_s[i];
  }
}

/******************************************************************************
 Functions
*`******************************************************************************/

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

