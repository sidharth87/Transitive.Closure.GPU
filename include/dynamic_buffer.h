#ifndef DYNAMIC_BUFFER_H
#define DYNAMIC_BUFFER_H

#include "sparse.h"

#define ORDERING    ROW_MAJOR

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
struct dynamic_buffer         //dynamic buffer
{
    cusp::array1d<INDEX_TYPE, MEM_TYPE> data_buffer;              //row sizes
    cusp::array1d<unsigned char, MEM_TYPE> status_buffer;              //row sizes
    
    int tuple_count;
    int tuple_size;
    int gap_count;

    size_t mem_size;        // total memory used
    size_t num_entries;     // number of currently filled entries

    dynamic_buffer()
    {}
    ~dynamic_buffer()
    {}

    // Resize the buffer using cusp resize
    void resize(const size_t n_tuples, const size_t n_gaps, const size_t tuple_size)
    {
    }

    // Check if resize is required
    // insert whenever possible, if not then append at the very end, and then sort the buffer
    void insert(const size_t n_tuples,  cusp::array1d<INDEX_TYPE, MEM_TYPE> &tuple)
    {
    }

    // based on first index of the predicate
    // p-ary search for multiple threads
    void select(INDEX_TYPE value_x)
    {
    }

    // based on first and second index of the predicate
    // p-ary search for multiple threads
    void select(INDEX_TYPE value_x, INDEX_TYPE value_y)
    {
    }
};


#include "primitives.h"
#include "scan.h"
#include "load.h"

#endif
