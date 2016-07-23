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

#define RESIZE_MULTIPLIER 1
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
    {
	is_sorted = 0;
	total_size = 0;
	used_size = 0;
    }
    
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

    void sort_and_remove_duplicates()
    {
	printf("[SORT Routine]\n");
	if (is_sorted == 0)
        {
            printf("[Before Buffer Size]: %d  %d \n", data_buffer[0].size(), data_buffer[1].size());
            switch(tuple_size)
            {
                case 1:
                thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[0].begin() )));
                break;

                case 2:
                thrust::sort_by_key(data_buffer[1].begin(), data_buffer[1].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[0].begin() )));
                output_dynamic_buffer("sort_1");
                thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[1].begin() )));
                output_dynamic_buffer("sort_2");
                break;

                //case 3:
                //thrust::sort_by_key(data_buffer[0].begin(), data_buffer[0].begin()+used_size, thrust::make_zip_iterator(thrust::make_tuple(data_buffer[1].begin(), data_buffer[2].begin() )));
                //break;

            }
	
            typedef thrust::device_vector<unsigned int>        IntVector;
            typedef IntVector::iterator                         IntIterator;
            typedef thrust::tuple< IntIterator, IntIterator >   IntIteratorTuple;
            typedef thrust::zip_iterator< IntIteratorTuple >    ZipIterator;

            ZipIterator newEnd = thrust::unique( thrust::make_zip_iterator( thrust::make_tuple( data_buffer[0].begin(), data_buffer[1].begin() ) ), thrust::make_zip_iterator( thrust::make_tuple( data_buffer[0].end(), data_buffer[1].end() ) ) );

            IntIteratorTuple endTuple = newEnd.get_iterator_tuple();

            data_buffer[0].erase( thrust::get<0>( endTuple ), data_buffer[0].end() );
            data_buffer[1].erase( thrust::get<1>( endTuple ), data_buffer[1].end() );
	    used_size = data_buffer[0].size();// - 1;
	    total_size = data_buffer[0].size();// - 1;
	    data_buffer[0].resize(used_size);
	    data_buffer[1].resize(used_size);
            printf("[After Buffer Size]: [%d] : %d %d \n", used_size, data_buffer[0].size() , data_buffer[1].size() );
            output_dynamic_buffer("sort_3");

            is_sorted = 1;
        }
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
	//sort_and_remove_duplicates();
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
	//sort_and_remove_duplicates();
    }

    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size>& buff, int offset)
    {
        printf("[B] In size = %d Out size = %d offset = %d total size = %d\n", used_size, buff.used_size, offset, total_size);
        if (used_size + buff.used_size - offset > total_size)
        {
            resize((used_size + buff.used_size - offset)*RESIZE_MULTIPLIER);
            total_size = (used_size + buff.used_size - offset)*RESIZE_MULTIPLIER;
        }
        printf("[A] In size = %d Out size = %d offset = %d total size = %d\n", used_size, buff.used_size, offset, total_size);
        
	for (int i = 0; i < tuple_size; i++)
	    device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&buff.data_buffer[i][offset]), used_size, buff.used_size - offset);

        used_size = used_size + buff.used_size - offset;
        is_sorted = 0;

	//sort_and_remove_duplicates();
    }


    void insert(const dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size>& buff, int offset, int size)
    {
        if (used_size + size > total_size)
        {
            resize((used_size + size)*RESIZE_MULTIPLIER);
            total_size = (used_size + size)*RESIZE_MULTIPLIER;
        }
        printf("[INSERT] In size = %d Out size = %d OFFSET %d\n", used_size, size, offset);

        for (int i = 0; i < tuple_size; i++)
            device::dynamic_buffer_insert<VALUE_TYPE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[i][0]), TPC(&buff.data_buffer[i][offset]), used_size, size);

        used_size = used_size + size;
        is_sorted = 0;
	//sort_and_remove_duplicates();
    }


    // based on first index of the predicate
    // p-ary search for multiple threads
    dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size> select(VALUE_TYPE value_x, int index_x, int select_size)
    {
	// search
        int h_lower_bound = 0, h_upper_bound = 0;
	
        int *g_lower_bound, *g_upper_bound;
        cudaMalloc(&g_lower_bound, sizeof(int));
        cudaMalloc(&g_upper_bound, sizeof(int));

	device::parySearchGPU<VALUE_TYPE, BLOCK_SIZE> <<<BLOCK_COUNT, BLOCK_SIZE>>> (TPC(&data_buffer[0][0]), /*used_size*/select_size, value_x, (g_lower_bound), (g_upper_bound));

        cudaMemcpy(&h_lower_bound, g_lower_bound, sizeof(int), cudaMemcpyDeviceToHost); 
        cudaMemcpy(&h_upper_bound, g_upper_bound, sizeof(int), cudaMemcpyDeviceToHost); 
        printf("Query range = %d %d\n", h_lower_bound, h_upper_bound);

        dynamic_buffer<VALUE_TYPE, MEM_TYPE, tuple_size> output(1, /*(h_upper_bound - h_lower_bound + 1)*/0, 0);

        if (h_lower_bound == -1 && h_upper_bound != -1)
          h_lower_bound = h_upper_bound;

        if (h_lower_bound != -1 && h_upper_bound == -1)
          h_upper_bound = h_lower_bound;

        if (h_lower_bound == -1 && h_upper_bound == -1)
          return output;

        output.insert(*this, h_lower_bound, (h_upper_bound - h_lower_bound + 1));
	//output.output_dynamic_buffer("XXXXXXXXXX");
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
