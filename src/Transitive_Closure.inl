#include <time.h>
#include <vector>

//void Transitive_closure(const std::string filename, int table_size1, int table_size2, int partition_size1, int partition_size2, int query_value)
void Transitive_closure(const std::string filename, int tuple_count)
{
	//set cuda device
	cudaSetDevice(0);

#if 0
	int dbuf1_size = table_size1;
	int dbuf2_size = table_size2;
        int dbuf1_partition = partition_size1;
        int dbuf2_partition = partition_size2;

	const size_t tuple_size = 2; 

	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf1 (0, dbuf1_size, dbuf1_size);
	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf2 (0, dbuf2_size, dbuf2_size);
	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> query_output_buffer;

        cusp::array1d<unsigned int, cusp::host_memory> cbuf1(dbuf1_size);
	cusp::array1d<unsigned int, cusp::host_memory> cbuf2(dbuf2_size);
        cusp::array1d<unsigned int, cusp::host_memory> cbuf1_rt(dbuf1_size);
	cusp::array1d<unsigned int, cusp::host_memory> cbuf2_rt(dbuf2_size);
	cusp::array1d<unsigned int, cusp::host_memory> output_buffer[tuple_size];

        int dbuf1_partition_size = dbuf1_size/dbuf1_partition;
	for (int i = 0; i < dbuf1_partition; i++)
        {
	  for (int j = 0; j < dbuf1_partition_size; j++)
 	  {
	    cbuf1[i*dbuf1_partition_size + j] = j + 123;
	    srand (time(NULL));
	    cbuf1_rt[i*dbuf1_partition_size + j] = 500;//rand() % 10 + 1;
	  }
        }

        int dbuf2_partition_size = dbuf2_size/dbuf2_partition;
	for (int i = 0; i < dbuf2_partition; i++)
        {
	  for (int j = 0; j < dbuf2_partition_size; j++)
	  {
	    cbuf2[i*dbuf2_partition_size + j] = j + 123;
	    srand (time(NULL));
	    cbuf2_rt[i*dbuf1_partition_size + j] = 51*j;//rand() % 10 + 1;
	  }
        }
  
	//for (int i = 0; i < tuple_size; i++)
	//{
	gbuf1.data_buffer[0] = cbuf1;
	gbuf2.data_buffer[0] = cbuf2;

	gbuf1.data_buffer[1] = cbuf1_rt;
	gbuf2.data_buffer[1] = cbuf2_rt;
	//}

	// Output the input buffers
	gbuf1.output_dynamic_buffer(filename+"_input_1");
	gbuf2.output_dynamic_buffer(filename+"_input_2");

	// insert gbuf2 into gbuf1
	gbuf1.insert(gbuf2);

	// Output gbuf1
	gbuf1.output_dynamic_buffer(filename+"_insert");
     

	// Query
        query_output_buffer = gbuf1.select(query_value, 0);

        printf("After sorting\n");
	gbuf1.output_dynamic_buffer(filename+"_sort");

        printf("Query output\n");
	query_output_buffer.output_dynamic_buffer(filename+"_query_output");
#endif

        cusp::array1d<unsigned int, cusp::host_memory> tl_index(tuple_count);
	cusp::array1d<unsigned int, cusp::host_memory> tr_index(tuple_count);

	const int tuple_size = 2;

   	FILE *fp;
	int in1, in2;
	int count = 0;
    	fp = fopen (filename.c_str(), "r"); 
        while (fscanf (fp, "%d\t%d\n",&in1, &in2) == 2 ) {

		tl_index[count] = in1;
		tr_index[count] = in2;
		//printf("Inpput ----> %d %d\n", in1, in2);
		count++;
        }
	fclose(fp);

	printf("Tuple count %d\n", tuple_count);
	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> input_buffer (0, tuple_count, tuple_count);
        input_buffer.data_buffer[0] = tl_index;
        input_buffer.data_buffer[1] = tr_index;

        spine<unsigned int, cusp::device_memory, tuple_size, 1> input_spine (input_buffer);
        spine <unsigned int, cusp::device_memory, tuple_size, 1> TC_spine (input_spine);
	TC_spine.sort_and_remove_duplicates();
        input_spine.output_spine(filename+"_IS");
        TC_spine.output_spine(filename+"_TCX");

              //spine <unsigned int, cusp::device_memory, tuple_size, 1> temp_spine;
	      //temp_spine = TC_spine.select(1, 0);

	int TC_count = tuple_count;
	int TC_prev_count = tuple_count;
	int BUCKET_COUNT = 1;

        //input_spine.output_spine(filename+"IS");
        //TC_spine.output_spine(filename+"_TC");

#if 0
	int key[BUCKET_COUNT];
	for (;;)
	{
 	  for(int i = 0; i < dbuffer_max_size; i++)
	  {
	    for (int j = 0; j < BUCKET_COUNT; j++)
	    {
	      if (i >= TC_spine.dbuffer_size[j])
	        key[j] = -1;
	      else
		key[j] = TC_spine.dbuffer[j].data_buffer[1][i];
	    }

            spine <unsigned int, cusp::device_memory, tuple_size, 1> temp_spine;
	    temp_spine = TC_spine.select_parallel(key, 0);
	    temp_spine.replace_index();
	    TC_spine.insert(temp_spine);
	    
	  }
	  TC_spine.sort_and_remove_duplicates(); 

	  if (TC_count == TC_prev_count)
	    break;

	}
#endif
#if 1

/*
      for (int k = 0; k < BUCKET_COUNT; k++)
      {
        cusp::array1d<unsigned int, cusp::host_memory> output_buffer[tuple_size];
        for (int i = 0; i < tuple_size; i++)
          output_buffer[i] = TC_spine.dbuffer[k].data_buffer[i];

        for (int i = 0; i < TC_spine.dbuffer[k].used_size; i++)
        {
          for (int j = 0; j < tuple_size; j++)
          {
            cout << output_buffer[j][i];
            if (j != tuple_size - 1)
              cout << "\t";
          }
          cout << "\n";
        }
      }
*/

	for (;;)
	//for (int y = 0; y < 2; y++)
	{
	  for (int i = 0; i < BUCKET_COUNT; i++)
	  {
            cusp::array1d<unsigned int, cusp::host_memory> output_buffer;
            cusp::array1d<unsigned int, cusp::host_memory> output_buffer2;
            //for (int i = 0; i < tuple_size; i++)
            output_buffer = TC_spine.dbuffer[i].data_buffer[1];
            output_buffer2 = TC_spine.dbuffer[i].data_buffer[0];

	    printf("Spine size [Before] [%d] ----- %d\n", i, TC_spine.dbuffer[i].used_size);
	    int current_spine_buffer_size = TC_spine.dbuffer[i].used_size;
	    for (int j = 0; j < current_spine_buffer_size; j++)
	    {
	      printf("Index = %d \n", output_buffer[j]);
              spine <unsigned int, cusp::device_memory, tuple_size, 1> temp_spine;
	      temp_spine = TC_spine.select(output_buffer[j], 0, current_spine_buffer_size);
	      //temp_spine = TC_spine.select(TC_spine.dbuffer[i].data_buffer[1][j], 0);

	      char name[512];
	      sprintf(name, "filename_%s_%d", filename.c_str(), j);
              temp_spine.output_spine(name);

	      temp_spine.replace_index(output_buffer2[j], 0);
              temp_spine.output_spine(name);
	      TC_spine.insert(temp_spine);
              TC_spine.output_spine(filename+"_TC1");
	    }
	    printf("Spine size [After] [%d] ----- %d\n", i, TC_spine.dbuffer[i].used_size);
	  }
	  TC_spine.sort_and_remove_duplicates();
          TC_spine.output_spine(filename+"_TC2");

	  	  
	  TC_count = 0;
	  for (int i = 0; i < BUCKET_COUNT; i++)
	    TC_count = TC_count + TC_spine.dbuffer[i].used_size;

          printf("TC_Count : TC_prev_count -- %d %d\n", TC_count, TC_prev_count);
	  if (TC_count == TC_prev_count)
	    break;
	  else
	    TC_prev_count = TC_count;
	 
	}
#endif
}
