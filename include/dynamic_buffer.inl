#ifndef DYNAMIC_BUFFER_DEVICE
#define DYNAMIC_BUFFER_DEVICE

#include<scan.h>

namespace device
{

template <typename VALUE_TYPE>
//__launch_bounds__(BLOCK_THREAD_SIZE,1)
__global__ void 
dynamic_buffer_insert(VALUE_TYPE *data_buff_out, const VALUE_TYPE *data_buff_in, size_t out_size, const size_t in_size)
{
    const int THREADS_PER_BLOCK = blockDim.x;
    const int thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const int grid_size  = THREADS_PER_BLOCK * gridDim.x;
    
    for (int i = thread_id; i < in_size; i = i + grid_size)
    {
    	data_buff_out[out_size + i] = data_buff_in[i];
    }
}


template <typename VALUE_TYPE, int THREADS_PER_BLOCK>
__global__ void 
parySearchGPU(VALUE_TYPE *data, int range_length, VALUE_TYPE search_keys, int* start_index, int* stop_index)
{
	//const int THREADS_PER_BLOCK = blockDim.x;
	__shared__ int cache[THREADS_PER_BLOCK+2];		// cache for boundary keys indexed by threadId
	__shared__ int cachel[THREADS_PER_BLOCK+2];		// cache for boundary keys indexed by threadId
	__shared__ int range_offset;					// index to subset for current iteration
        //int local_result;
        int range_length_copy = range_length;

	int sk, old_range_length=range_length,range_start;
	// initialize search range using a single thread
	if (threadIdx.x==0) 
	{
		range_offset=0;
		// cache search key and upper bound in shared memory
		cache[THREADS_PER_BLOCK]= 0x7FFFFFFF;
		cache[THREADS_PER_BLOCK+1]= search_keys;//search_keys[blockIdx.x];
	}

	// require a sync, since each thread is going to read the above now
	syncthreads();

	sk = cache[THREADS_PER_BLOCK+1];

	while (range_length>THREADS_PER_BLOCK)
	{
		range_length = range_length/THREADS_PER_BLOCK;
		// check for division underflow
		if (range_length * THREADS_PER_BLOCK < old_range_length)
			range_length+=1;
		old_range_length=range_length;

		// cache the boundary keys
		range_start = range_offset + threadIdx.x * range_length;
		cache[threadIdx.x]=data[range_start];
		syncthreads();

		// if the seached key is within this thread’s subset,
		// make it the one for the next iteration
		//printf("[%d] --> %d (%d %d)\n", threadIdx.x, sk, cache[threadIdx.x], cache[threadIdx.x+1]);
		if (sk>=cache[threadIdx.x] && sk<cache[threadIdx.x+1])
		{
			printf("[HIT U %d] --> %d (%d %d)\n", threadIdx.x, sk, cache[threadIdx.x], cache[threadIdx.x+1]);
			range_offset = range_start;
		}

		// all threads need to start next iteration with the new subset
		syncthreads();
	}

	// store search result
	range_start = range_offset + threadIdx.x;
	if (sk==data[range_start] && sk < data[range_start+1])
	{
	    //local_result++;
            *stop_index = range_start;
	    //printf("Upper [%d] ----> %d\n", threadIdx.x, *stop_index);
	    //printf("[%d] Upper [%d] ----> %d\n", range_offset, threadIdx.x, range_start);
	}


        range_length = range_length_copy;
        old_range_length = range_length_copy;
        range_start = 0;
	
        if (threadIdx.x==0) 
	{
		range_offset=range_length - 1;
		for (int i = 0; i < THREADS_PER_BLOCK+2; i++)
		  cachel[i] = 0;
		// cache search key and upper bound in shared memory
		cachel[THREADS_PER_BLOCK]= 0x80000000;
		cachel[THREADS_PER_BLOCK+1]= search_keys;//search_keys[blockIdx.x];
	}

	// require a sync, since each thread is going to read the above now
	syncthreads();
	sk = cachel[THREADS_PER_BLOCK+1];
        while (range_length>THREADS_PER_BLOCK)
        {
                range_length = range_length/THREADS_PER_BLOCK;
                // check for division underflow
                if (range_length * THREADS_PER_BLOCK < old_range_length)
                        range_length+=1;
                old_range_length=range_length;

                // cache the boundary keys
                range_start = range_offset - threadIdx.x * range_length;
                cachel[threadIdx.x]=data[range_start];
                syncthreads();

                // if the seached key is within this thread’s subset,
                // make it the one for the next iteration
                //if (sk<=cache[threadIdx.x] && sk>cache[threadIdx.x-1])
                //if (sk>cache[threadIdx.x] && sk<=cache[threadIdx.x+1])
                if (sk>cachel[threadIdx.x + 1] && sk<=cachel[threadIdx.x])
		{
			printf("[HIT L %d] --> %d (%d %d)\n", threadIdx.x, sk, cachel[threadIdx.x], cachel[threadIdx.x+1]);
                        range_offset = range_start;
		}

                // all threads need to start next iteration with the new subset
                syncthreads();
        }

#if 1
        // store search result
        range_start = range_offset - threadIdx.x;
        if (sk==data[range_start] && sk > data[range_start-1])
        {
            *start_index = range_start;
            printf("[%d] Lower [%d] ----> %d\n", range_offset, threadIdx.x, range_start);
            //printf("Lower [%d] ----> %d\n", threadIdx.x, *start_index);
        }
#endif

        
}

}       //namespace device

#endif
