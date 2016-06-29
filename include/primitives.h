#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "matrix_info.h"
#include "primitives_device.h"

#include "sparse_update.inl"

namespace device
{

/////////////////////////////////////////////////////////////////////////
/////////////////  Entry Wrapper Functions  /////////////////////////////
/////////////////////////////////////////////////////////////////////////
template <typename INDEX_TYPE, typename VALUE_TYPE>
void count_sorted_indices(	const cusp::array1d<INDEX_TYPE, cusp::device_memory> indices,
                   			cusp::array1d<INDEX_TYPE, cusp::device_memory> count,
                   			const int size)
{
    const INDEX_TYPE * I = thrust::raw_pointer_cast(&indices[0]);
    const INDEX_TYPE * X = thrust::raw_pointer_cast(&count[0]);

    if(size == 0)
    {
        // empty matrix
        return;
    }
    else if(size < static_cast<size_t>(WARP_SIZE))
    {
        // small matrix
        count_sorted_indices_serial_kernel<INDEX_TYPE,VALUE_TYPE> <<<1,1>>> (size, I, X);
        return;
    }

    const unsigned int BLOCK_SIZE = 256;
    const unsigned int MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks(count_sorted_indices_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE>, BLOCK_SIZE, (size_t) 0);
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    const unsigned int num_units  = size / WARP_SIZE;
    const unsigned int num_warps  = std::min(num_units, WARPS_PER_BLOCK * MAX_BLOCKS);
    const unsigned int num_blocks = cusp::system::cuda::DIVIDE_INTO(num_warps, WARPS_PER_BLOCK);
    const unsigned int num_iters  = cusp::system::cuda::DIVIDE_INTO(num_units, num_warps);
    
    const unsigned int interval_size = WARP_SIZE * num_iters;

    const INDEX_TYPE tail = num_units * WARP_SIZE; // do the last few nonzeros separately (fewer than WARP_SIZE elements)

    const unsigned int active_warps = (interval_size == 0) ? 0 : cusp::system::cuda::DIVIDE_INTO(tail, interval_size);

    cusp::array1d<INDEX_TYPE,cusp::device_memory> temp_rows(active_warps);
    cusp::array1d<VALUE_TYPE,cusp::device_memory> temp_vals(active_warps);

    count_sorted_indices_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE> <<<num_blocks, BLOCK_SIZE>>>
        (tail, interval_size, I, X,
         thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]));

    count_sorted_indices_update_kernel<INDEX_TYPE, VALUE_TYPE, BLOCK_SIZE> <<<1, BLOCK_SIZE>>>
        (active_warps, thrust::raw_pointer_cast(&temp_rows[0]), thrust::raw_pointer_cast(&temp_vals[0]), X);
    
    count_sorted_indices_serial_kernel<INDEX_TYPE,VALUE_TYPE> <<<1,1>>>
        (size - tail, I + tail, X);
}

} //namespace device

#endif
