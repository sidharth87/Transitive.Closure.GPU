#ifndef DYNAMIC_BUFFER_DEVICE
#define DYNAMIC_BUFFER_DEVICE


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


__global__ void 
dynamic_buffer_serch()
{

}



template <typename VALUE_TYPE>
__global__ void 
parySearchGPU(VALUE_TYPE ∗data, int range_length, int∗ search_keys, int∗ results)
{
	const int THREADS_PER_BLOCK = blockDim.x;
	shared int cache[THREADS_PER_BLOCK+2];		// cache for boundary keys indexed by threadId
	shared int range_offset;					// index to subset for current iteration

	int sk, old_range_length=range_length,range_start;
	// initialize search range using a single thread
	if (threadIdx.x==0) 
	{
		range_offset=0;
		// cache search key and upper bound in shared memory
		cache[THREADS_PER_BLOCK]= 0x7FFFFFFF;
		cache[THREADS_PER_BLOCK+1]= search_keys[blockIdx.x];
	}

	// require a sync, since each thread is going to read the above now
	syncthreads();

	sk = cache[THREADS_PER_BLOCK+1];
	while (range_length>THREADS_PER_BLOCK)
	{
		range_length = range_length/THREADS_PER_BLOCK;
		// check for division underflow
		if (range_length ∗ THREADS_PER_BLOCK < old_range_length)
			range_length+=1;
		old_range_length=range_length;

		// cache the boundary keys
		range_start = range_offset + threadIdx.x ∗ range_length;
		cache[threadIdx.x]=data[range_start];
		syncthreads();

		// if the seached key is within this thread’s subset,
		// make it the one for the next iteration
		if (sk>=cache[threadIdx.x] && sk<cache[threadIdx.x+1])
			range_offset = range_start;

		// all threads need to start next iteration with the new subset
		syncthreads();
	}

	// store search result
	range_start = range_offset + threadIdx.x;
	if (sk==data[range_start])
		results[blockIdx.x]=range_start;
}




}       //namespace device

#endif
