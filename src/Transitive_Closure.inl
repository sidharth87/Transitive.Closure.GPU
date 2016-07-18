void Transitive_closure(const std::string filename, int table_size1, int table_size2, int partition_size1, int partition_size2, int query_value)
{
	//set cuda device
	cudaSetDevice(0);

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
	cusp::array1d<unsigned int, cusp::host_memory> output_buffer[tuple_size];

        int dbuf1_partition_size = dbuf1_size/dbuf1_partition;
	for (int i = 0; i < dbuf1_partition; i++)
        {
	  for (int j = 0; j < dbuf1_partition_size; j++)
	    cbuf1[i*dbuf1_partition_size + j] = j;
        }

        int dbuf2_partition_size = dbuf2_size/dbuf2_partition;
	for (int i = 0; i < dbuf2_partition; i++)
        {
	  for (int j = 0; j < dbuf2_partition_size; j++)
	    cbuf2[i*dbuf2_partition_size + j] = j;
        }
  
	for (int i = 0; i < tuple_size; i++)
	{
	   gbuf1.data_buffer[i] = cbuf1;
	   gbuf2.data_buffer[i] = cbuf2;
	}

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
}
