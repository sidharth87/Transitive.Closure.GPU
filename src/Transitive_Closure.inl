void Transitive_closure(const std::string &filename)
{
	//set cuda device
	cudaSetDevice(0);

	int dbuf1_size = 256;
	int dbuf2_size = 256;
        int dbuf1_partition = 32;
        int dbuf2_partition = 32;

	const size_t tuple_size = 1; 

	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf1;
	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf2;
	dynamic_buffer<unsigned int, cusp::host_memory, tuple_size> query_output_buffer;

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

	gbuf1.is_sorted = 0;
	gbuf1.total_size = dbuf1_size;
	gbuf1.used_size = dbuf1_size;

	gbuf2.is_sorted = 0;
	gbuf2.total_size = dbuf2_size;
	gbuf2.used_size = dbuf2_size;

	gbuf1.insert(gbuf2);

	for (int i = 0; i < tuple_size; i++)
	{
		output_buffer[i] = gbuf1.data_buffer[i];
		for (int j = 0; j < gbuf1.used_size; j++)
			printf("output buffer %d = %d\n", j, output_buffer[i][j]);
	}
     
        query_output_buffer = gbuf1.select(2, 0);
        printf("After sorting\n");
	for (int i = 0; i < tuple_size; i++)
	{
		output_buffer[i] = gbuf1.data_buffer[i];
		for (int j = 0; j < gbuf1.used_size; j++)
			printf("output buffer %d = %d\n", j, output_buffer[i][j]);
	}

        printf("Query output\n");
	for (int i = 0; i < tuple_size; i++)
	{
		for (int j = 0; j < query_output_buffer.used_size; j++)
			printf("Query output buffer %d = %d\n", j, query_output_buffer.data_buffer[i][j]);
	}
}
