#ifndef SPINE_H
#define SPINE_H

#include "sparse.h"
#include "dynamic_buffer.h"

#define ORDERING    ROW_MAJOR

template <typename VALUE_TYPE, typename MEM_TYPE, size_t TUPLE_SIZE, size_t BUCKET_COUNT>
struct spine         //spine
{
    cusp::array1d<dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE>, MEM_TYPE> dbuffer[BUCKET_COUNT];              //row sizes
    size_t used_size;    // number of currently filled entries

    spine()
    {}

    ~spine()
    {}

    // Check if resize is required
    // group based the tupples based on the first index, so they map to the same hash index, and can be inserted as a group
    // use the hash function to find the right index and invoke the insert of dynaimc buffer 
    void insert(const size_t n_tuples,  cusp::array1d<VALUE_TYPE, MEM_TYPE> &tuple)
    {
    }

    void insert(spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT>& input_spine)
    {
	for (int i = 0; i < BUCKET_COUNT; i++)
        {
          dbuffer[i].insert(input_spine.dbuffer[i]);
        }
    }

    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE>& buff)
    {
	if (buff.is_sorted == 0)
        {
            switch(TUPLE_SIZE)
            {
                case 1:
                thrust::sort_by_key(buff.data_buffer[0].begin(), buff.data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(buff.data_buffer[0].begin() )));
                break;

                case 2:
                thrust::sort_by_key(buff.data_buffer[0].begin(), buff.data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(buff.data_buffer[1].begin() )));
                break;

                case 3:
                thrust::sort_by_key(buff.data_buffer[0].begin(), buff.data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(buff.data_buffer[1].begin(), buff.data_buffer[2].begin() )));
                break;

            }
            buff.is_sorted = 1;
        }

      //spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT> temp_spine;
      int insert_buffer_count[BUCKET_COUNT] = {0};
      for (int i = 0; i < buff.used_size; i++)
	insert_buffer_count[hash(buff.data_buffer[0][i])]++;
	
      for (int i = 0; i < BUCKET_COUNT; i++)
      {
	if (i == 0)
          dbuffer[i].insert(buff, 0);
	else
          dbuffer[i].insert(buff, insert_buffer_count[i - 1]);
      }
    }

    // based on first index of the predicate
    // use the hash function to identify the bin, and invoke select of dynamic buffer
    spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT>& select(VALUE_TYPE value_x, int index_x )
    {
	spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT> output;
        for (int i = 0; i < BUCKET_COUNT; i++)
	  output.dbuffer[i] = dbuffer[i].select(value_x, index_x);

	return output;
    }


    dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE>& select_dynamic_buffer(VALUE_TYPE value_x, int index_x )
    {
        dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE> output;
        for (int i = 0; i < BUCKET_COUNT; i++)
          output.insert(dbuffer[i].select(value_x, index_x));

        return output;
    }


    // based on first and second index of the predicate
    // use the hash function to identify the bin, and invoke select of dynamic buffer
    void select(VALUE_TYPE value_x, VALUE_TYPE value_y)
    {
    }

    int hash(VALUE_TYPE value_x)
    {
	return value_x % BUCKET_COUNT;
    }
};


#include "primitives.h"
#include "scan.h"
#include "load.h"

#endif
