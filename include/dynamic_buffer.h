#ifndef DYNAMIC_BUFFER_H
#define DYNAMIC_BUFFER_H

#include <iostream>
#include <fstream>
#include <string>
#include "sparse.h"
#include "dynamic_buffer.inl"

using namespace std;

#define BLOCK_SIZE 32
#define BLOCK_COUNT 1
#define TPC(x) thrust::raw_pointer_cast(x)

#define RESIZE_MULTIPLIER 1.5
#define ORDERING    ROW_MAJOR

template <typename VALUE_TYPE, typename MEM_TYPE, size_t tuple_size>
struct dynamic_buffer         //dynamic buffer
{
    cusp::array1d<VALUE_TYPE, MEM_TYPE> data_buffer[tuple_size];              //row sizes
    //cusp::array1d<unsigned char, MEM_TYPE> status_buffer;              //row sizes

    int is_sorted;
    size_t total_size;      // How many total entries are there in data_buffer
    size_t used_size;       // How many entries are filled with values
    //size_t tuple_size;      // number of variables/columns in the predicate/in each tuple

    dynamic_buffer()
    {}
    
    dynamic_buffer(int sort_status, size_t tsize, int usize)
    {
	is_sorted = sort_status;
	total_size = tsize;
	used_size = usize;
    }
    
    ~dynamic_buffer()
    {}


    // writes to std out
    void print_dynamic_buffer()
    {
      /*
      cusp::array1d<VALUE_TYPE, cusp::host_memory> output_buffer[tuple_size];
      for (int i = 0; i < tuple_size; i++)
      {
	output_buffer[i] = data_buffer[i];
	for (int j = 0; j < used_size; j++)
	{
	  cout << output_buffer[i][j] << "\t";
	}
	cout << "\n";
      }
      */
    }

    // writes to std out
    void output_dynamic_buffer(string filename)
    {
      ofstream myfile;
      myfile.open (filename.c_str());
      cusp::array1d<VALUE_TYPE, cusp::host_memory> output_buffer[tuple_size];

      for (int i = 0; i < tuple_size; i++)
	output_buffer[i] = data_buffer[i];

      for (int i = 0; i < used_size; i++)
      {
	for (int j = 0; j < tuple_size; j++)
	{
	  myfile << output_buffer[j][i];
	  if (j != tuple_size - 1)
	    myfile << "\t";
	}
	myfile << "\n";
      }
      myfile.close();
    }

    // Resize the buffer using cusp resize
    void resize(const size_t new_size)
    {
       for (int i = 0; i < tuple_size; i++)
           data_buffer[i].resize(new_size);
    }

    // Check if resize is required
    // insert whenever possible, if not then append at the very end, and then sort the buffer
    void insert(const size_t n_tuples,  cusp::array1d<VALUE_TYPE, MEM_TYPE> *tuple)
    {
        if (n_tuples + used_size > total_size)
        {
            resize((used_size + n_tuples)*RESIZE_MULTIPLIER);
            total_size = (used_size + n_tuples)*RESIZE_MULTIPLIER;
        }

	for (int i = 0; i < tuple_size; i++)
	    device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&tuple[i][0]), used_size, n_tuples);

        used_size = used_size + n_tuples;
        is_sorted = 0;
    }
    
    // Check if resize is required first
    // Concatenate new tuples onto the end of data_buffer (sorting is done lazily)
    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size>& buff)
    {
        if (used_size + buff.used_size > total_size)
        {
            resize((used_size + buff.used_size)*RESIZE_MULTIPLIER);
            total_size = (used_size + buff.used_size)*RESIZE_MULTIPLIER;
        }
        printf("In size = %d Out size = %d\n", used_size, buff.used_size);       
        
	for (int i = 0; i < tuple_size; i++)
	    device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&buff.data_buffer[i][0]), used_size, buff.used_size);

        used_size = used_size + buff.used_size;
        is_sorted = 0;
    }

    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size>& buff, int offset)
    {
        if (used_size + buff.used_size - offset > total_size)
        {
            resize((used_size + buff.used_size - offset)*RESIZE_MULTIPLIER);
            total_size = (used_size + buff.used_size - offset)*RESIZE_MULTIPLIER;
        }
        printf("In size = %d Out size = %d\n", used_size, buff.used_size);       
        
	for (int i = 0; i < tuple_size; i++)
	    device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&buff.data_buffer[i][offset]), used_size, buff.used_size - offset);

        used_size = used_size + buff.used_size - offset;
        is_sorted = 0;
    }


    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size>& buff, int offset, int size)
    {
        if (used_size + size > total_size)
        {
            resize((used_size + size)*RESIZE_MULTIPLIER);
            total_size = (used_size + size)*RESIZE_MULTIPLIER;
        }
        printf("In size = %d Out size = %d\n", used_size, buff.used_size);

        for (int i = 0; i < tuple_size; i++)
            device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&buff.data_buffer[i][offset]), used_size, size);

        used_size = used_size + size;
        is_sorted = 0;
    }


    // based on first index of the predicate
    // p-ary search for multiple threads
    dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size> select(VALUE_TYPE value_x, int index_x)
    {
	if (is_sorted == 0)
	{
	    switch(tuple_size)
	    {
		case 1:
  		thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[0].begin() )));
		break;

		case 2:
		thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[1].begin() )));
		break;

		//case 3:
		//thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[1].begin(), data_buffer[2].begin() )));
		//break;

	    }
	    is_sorted = 1;
	}
	// search
        int h_lower_bound = 0, h_upper_bound = 0;
	
        int *g_lower_bound, *g_upper_bound;
        cudaMalloc(&g_lower_bound, sizeof(int));
        cudaMalloc(&g_upper_bound, sizeof(int));

	device::parySearchGPU<VALUE_TYPE, BLOCK_SIZE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[0][0]), used_size, value_x, (g_lower_bound), (g_upper_bound));

        cudaMemcpy(&h_lower_bound, g_lower_bound, sizeof(int), cudaMemcpyDeviceToHost); 
        cudaMemcpy(&h_upper_bound, g_upper_bound, sizeof(int), cudaMemcpyDeviceToHost); 
        printf("Query range = %d %d\n", h_lower_bound, h_upper_bound);
#if 1

        dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size> output(1, (h_upper_bound - h_lower_bound + 1), (h_upper_bound - h_lower_bound + 1));
        output.insert(*this, h_lower_bound, (h_upper_bound - h_lower_bound + 1));
        /*
        for (int i = 0; i < tuple_size; i++)
        {
          cusp::array1d<VALUE_TYPE, cusp::host_memory> temp(h_upper_bound - h_lower_bound + 1);
          for (int j = 0; j < h_upper_bound - h_lower_bound + 1; j++)
          {
            temp[j] = data_buffer[i][j + h_lower_bound];
          }
          output.data_buffer[i] = temp;
        }
        */
#endif	
	return output;
    }

    // based on first and second index of the predicate
    // p-ary search for multiple threads
    void select(VALUE_TYPE value_x, int index_x, VALUE_TYPE value_y, int index_y)
    {
    }
};


#include "primitives.h"
#include "scan.h"
#include "load.h"

#endif
