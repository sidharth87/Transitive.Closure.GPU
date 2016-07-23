#ifndef SPINE_H
#define SPINE_H

#include "sparse.h"
#include "dynamic_buffer.h"
#include "spine.inl"

template <typename VALUE_TYPE, typename MEM_TYPE, size_t TUPLE_SIZE, size_t BUCKET_COUNT>
struct spine         //spine
{
    //cusp::array1d<dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE>, MEM_TYPE> dbuffer[BUCKET_COUNT];              //row sizes
    dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE> dbuffer[BUCKET_COUNT];              //row sizes

    size_t dbuffer_size[BUCKET_COUNT];
    size_t dbuffer_max_size;

    //size_t used_size;    // number of currently filled entries

    spine()
    {
	//used_size = 0;
	dbuffer_max_size = 0;
    }

    ~spine()
    {}

#if 1
    spine(spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT>& input_spine)
    {
      for (int i = 0; i < BUCKET_COUNT; i++)
      {
	  dbuffer[i].used_size = 0;//input_spine.dbuffer[i].used_size;
	  dbuffer[i].total_size = 0;//input_spine.dbuffer[i].total_size;
          dbuffer[i].insert(input_spine.dbuffer[i]);
      }
    }

    spine(dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE>& buff)
    {
      printf("[Constructor] Buff size = %d\n", buff.used_size);
      
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

    // Check if resize is required
    // group based the tupples based on the first index, so they map to the same hash index, and can be inserted as a group
    // use the hash function to find the right index and invoke the insert of dynaimc buffer 
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
	buff.sort_and_remove_duplicates();

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
    // Version 1
    spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT> select(VALUE_TYPE value_x, int index_x, int size )
    {
	spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT> output;
        for (int i = 0; i < BUCKET_COUNT; i++)
	  output.dbuffer[i] = dbuffer[i].select(value_x, index_x, size);

	output.print_spine();
	return output;
    }

#if 0
    // Version 2
    spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT>& select_parallel(VALUE_TYPE* value_x, int index_x, int bucket_index )
    {
	spine<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE, BUCKET_COUNT> output;
        //for (int i = 0; i < BUCKET_COUNT; i++)
	//  output.dbuffer[i] = dbuffer[i].select(value_x, index_x);
	//
	//return output;

	cusp::array1d<unsigned int, cusp::device_memory> start_index(BUCKET_COUNT);
	cusp::array1d<unsigned int, cusp::device_memory> stop_index(BUCKET_COUNT);

	int h_lower_bound = 0, h_upper_bound = 0;

        device::spineparySearchGPU<VALUE_TYPE, BLOCK_SIZE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&(dbuffer[bucket_index].data_buffer[0][0])), dbuffer[bucket_index].used_size, value_x, start_index, stop_index);

        /*
        cudaMemcpy(&h_lower_bound, g_lower_bound, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_upper_bound, g_upper_bound, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Query range = %d %d\n", h_lower_bound, h_upper_bound);

        dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size> output(1, 0, 0);

        if (h_lower_bound == -1 && h_upper_bound == -1)
          return output;

        output.insert(*this, h_lower_bound, (h_upper_bound - h_lower_bound + 1));
        return output;
	*/
	return NULL;
    }
#endif

    void replace_index(VALUE_TYPE value_x, int index)
    {
        for (int i = 0; i < BUCKET_COUNT; i++)
        {
          device::spine_replace<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&(dbuffer[i].data_buffer[index][0])), value_x, dbuffer[i].used_size);
	}
    }


    void output_spine(string filename)
    {
      ofstream myfile;
      myfile.open (filename.c_str());
      
      for (int k = 0; k < BUCKET_COUNT; k++)
      {
        cusp::array1d<VALUE_TYPE, cusp::host_memory> output_buffer[TUPLE_SIZE];

        for (int i = 0; i < TUPLE_SIZE; i++)
          output_buffer[i] = dbuffer[k].data_buffer[i];

        for (int i = 0; i < dbuffer[k].used_size; i++)
        {
          for (int j = 0; j < TUPLE_SIZE; j++)
          {
            myfile << output_buffer[j][i];
            if (j != TUPLE_SIZE - 1)
              myfile << "\t";
          }
          myfile << "\n";
        }
      }
      myfile.close();
    }


    void print_spine()
    {
      
      for (int k = 0; k < BUCKET_COUNT; k++)
      {
        cusp::array1d<VALUE_TYPE, cusp::host_memory> output_buffer[TUPLE_SIZE];

        for (int i = 0; i < TUPLE_SIZE; i++)
          output_buffer[i] = dbuffer[k].data_buffer[i];

        for (int i = 0; i < dbuffer[k].used_size; i++)
        {
          for (int j = 0; j < TUPLE_SIZE; j++)
          {
            cout << output_buffer[j][i];
            if (j != TUPLE_SIZE - 1)
              cout << "\t";
          }
          cout << "\n";
        }
      }
    }

    void sort_and_remove_duplicates()
    {
        for (int i = 0; i < BUCKET_COUNT; i++)
          dbuffer[i].sort_and_remove_duplicates();
    }


    dynamic_buffer<VALUE_TYPE, MEM_TYPE, TUPLE_SIZE> select_dynamic_buffer(VALUE_TYPE value_x, int index_x )
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
#endif
};


//#include "primitives.h"
//#include "scan.h"
//#include "load.h"

#endif
