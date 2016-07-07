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

}       //namespace device

#endif
