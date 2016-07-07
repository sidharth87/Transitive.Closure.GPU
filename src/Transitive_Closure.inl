void Transitive_closure(const std::string &filename)
{
	//set cuda device
	cudaSetDevice(0);

	int dbuf1_size = 256;
	int dbuf2_size = 512;

	dynamic_buffer<unsigned int, cusp::device_memory> gbuf1;
	dynamic_buffer<unsigned int, cusp::device_memory> gbuf2;

        cusp::array1d<unsigned int, cusp::host_memory> cbuf1(dbuf1_size);
	cusp::array1d<unsigned int, cusp::host_memory> cbuf2(dbuf2_size);

	cusp::array1d<unsigned int, cusp::host_memory> output_buffer;

	for (int i = 0; i < dbuf1_size; i++)
	  cbuf1[i] = i;

	for (int i = 0; i < dbuf2_size; i++)
	  cbuf2[i] = dbuf1_size + i;

	gbuf1.data_buffer = cbuf1;
	gbuf2.data_buffer = cbuf2;

	gbuf1.is_sorted = 0;
	gbuf1.total_size = dbuf1_size;
	gbuf1.used_size = dbuf1_size;
        gbuf1.tuple_size = 1;

	gbuf2.is_sorted = 0;
	gbuf2.total_size = dbuf2_size;
	gbuf2.used_size = dbuf2_size;
        gbuf2.tuple_size = 1;

	gbuf1.insert(gbuf2);

	output_buffer = gbuf1.data_buffer;

	for (int i = 0; i < gbuf1.used_size; i++)
	  printf("output buffer %d = %d\n", i, output_buffer[i]);

}




