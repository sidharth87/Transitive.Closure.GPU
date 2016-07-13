void Transitive_closure(const std::string &filename)
{
	//set cuda device
	cudaSetDevice(0);

	int dbuf1_size = 256;
	int dbuf2_size = 256;
	const size_t tuple_size = 1; 

	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf1;
	dynamic_buffer<unsigned int, cusp::device_memory, tuple_size> gbuf2;

        cusp::array1d<unsigned int, cusp::host_memory> cbuf1(dbuf1_size);
	cusp::array1d<unsigned int, cusp::host_memory> cbuf2(dbuf2_size);

	cusp::array1d<unsigned int, cusp::host_memory> output_buffer[tuple_size];

	for (int i = 0; i < dbuf1_size; i++)
	  cbuf1[i] = dbuf1_size + dbuf2_size - i;

	for (int i = 0; i < dbuf2_size; i++)
	  cbuf2[i] = dbuf2_size - i;

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
     
        gbuf1.select(5, 0);
        printf("After sorting\n");
	for (int i = 0; i < tuple_size; i++)
	{
		output_buffer[i] = gbuf1.data_buffer[i];
		for (int j = 0; j < gbuf1.used_size; j++)
			printf("output buffer %d = %d\n", j, output_buffer[i][j]);
	}
}
