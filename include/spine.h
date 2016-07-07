#ifndef SPINE_H
#define SPINE_H

#include "sparse.h"
#include "dynamic_buffer.h"

#define ORDERING    ROW_MAJOR

template <typename INDEX_TYPE, typename VALUE_TYPE, typename MEM_TYPE, size_t BINS>
struct spine         //spine
{
    cusp::array1d<dynamic_buffer<VALUE_TYPE, MEM_TYPE>*, MEM_TYPE> buffer_ptr;              //row sizes
    int size;
    size_t num_entries;     // number of currently filled entries

    spine()
    {}
    ~spine()
    {}

    // Resize the buffer using cusp resize
    void resize(const size_t n_tuples, const size_t n_gaps, const size_t tuple_size)
    {
    }

    // Check if resize is required
    // group based the tupples based on the first index, so they map to the same hash index, and can be inserted as a group
    // use the hash function to find the right index and invoke the insert of dynaimc buffer 
    void insert(const size_t n_tuples,  cusp::array1d<INDEX_TYPE, MEM_TYPE> &tuple)
    {
    }

    // based on first index of the predicate
    // use the hash function to identify the bin, and invoke select of dynamic buffer
    void select(INDEX_TYPE value_x)
    {
    }

    // based on first and second index of the predicate
    // use the hash function to identify the bin, and invoke select of dynamic buffer
    void select(INDEX_TYPE value_x, INDEX_TYPE value_y)
    {
    }

    int hash(INDEX_TYPE value_x)
    {
    }
};


#include "primitives.h"
#include "scan.h"
#include "load.h"

#endif
