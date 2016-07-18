#include<stdio.h>
#include<stdlib.h>
#include "Matrix_Test.h"
#include "load_matrix.h"
#include "dynamic_buffer.h"
#include "spine.h"

void Transitive_closure(const std::string &filename);
void Matrix_Test(const std::string filename);
void createStreams();
void createStreams(const int ID);
void destroyStreams();
void destroyStreams(const int ID);

#include "Transitive_Closure.inl"

void createStreams(const int ID)
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamCreate(&__multiStreams[ID][i]);
}

void destroyStreams(const int ID)
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamDestroy(__multiStreams[ID][i]);
}

void createStreams()
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamCreate(&__streams[i]);
}

void destroyStreams()
{
	for(int i=0; i<NUM_STREAMS; i++)
		cudaStreamDestroy(__streams[i]);
}

void Matrix_Test(const std::string filename, int table1_size, int table2_size, int ps1, int ps2, int q)
{
	#if(MULTI_GPU == 1)
		//FillTests(filename);
	#else
		Transitive_closure(filename, table1_size, table2_size, ps1, ps2, q);
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//	Parse input file and run test
////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char** argv)
{
	if(argc != 7)
	{
		fprintf(stderr, "Invalid input...\n");
		fprintf(stderr, "Usage: TC <filename template> table_size1 table_size2 partiiton_size1 partition_size2 query_value\n");
		exit(1);
	}

	std::string filename(argv[1]);
	Transitive_closure(filename, atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
}

int main(int argc, char **argv)
{
	fprintf(stderr, "TEST START\n");
	runTest(argc, argv);
	fprintf(stderr, "TEST COMPLETE\n");
	return 0;
}
