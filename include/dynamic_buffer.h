#ifndef DYNAMIC_BUFFER_H
#define DYNAMIC_BUFFER_H

#include "sparse.h"
#include "dynamic_buffer.inl"

#define BLOCK_SIZE 512
#define BLOCK_COUNT 64
#define TPC(x) thrust::raw_pointer_cast(x)

#define RESIZE_MULTIPLIER 1.5
#define ORDERING    ROW_MAJOR

template <typename VALUE_TYPE, typename MEM_TYPE>
struct dynamic_buffer         //dynamic buffer
{
    cusp::array1d<VALUE_TYPE, MEM_TYPE> data_buffer;              //row sizes
    //cusp::array1d<unsigned char, MEM_TYPE> status_buffer;              //row sizes

    unsigned char is_sorted;
    size_t total_size;      // How many total entries are there in data_buffer
    size_t used_size;       // How many entries are filled with values
    size_t tuple_size;      // number of variables/columns in the predicate/in each tuple

    dynamic_buffer()
    {}
    
    ~dynamic_buffer()
    {}

    // Resize the buffer using cusp resize
    void resize(const size_t new_size)
    {
       data_buffer.resize(new_size);
    }

    // Check if resize is required
    // insert whenever possible, if not then append at the very end, and then sort the buffer
    void insert(const size_t n_tuples,  cusp::array1d<VALUE_TYPE, MEM_TYPE> &tuple)
    {
      
    }
    
    // Check if resize is required first
    // Concatenate new tuples onto the end of data_buffer (sorting is done lazily)
    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE>& buff)
    {
        
        if (used_size + buff.used_size > total_size)
        {
            resize((used_size + buff.used_size)*RESIZE_MULTIPLIER);
            total_size = (used_size + buff.used_size)*RESIZE_MULTIPLIER;
        }
        printf("In size = %d Out size = %d\n", used_size, buff.used_size);       
        device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[0]), TPC(&buff.data_buffer[0]), used_size, buff.used_size);

        used_size = used_size + buff.used_size;
    }

    // based on first index of the predicate
    // p-ary search for multiple threads
    void select(VALUE_TYPE value_x)
    {
    }

    // based on first and second index of the predicate
    // p-ary search for multiple threads
    void select(VALUE_TYPE value_x, VALUE_TYPE value_y)
    {
    }
};


#include "primitives.h"
#include "scan.h"
#include "load.h"

#endif
