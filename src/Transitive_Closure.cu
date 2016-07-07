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

void Matrix_Test(const std::string filename)
{
	#if(MULTI_GPU == 1)
		//FillTests(filename);
	#else
		Transitive_closure(filename);
	#endif
}

////////////////////////////////////////////////////////////////////////////////
//	Parse input file and run test
////////////////////////////////////////////////////////////////////////////////

void runTest(int argc, char** argv)
{
	if(argc != 2)
	{
		fprintf(stderr, "Invalid input...\n");
		fprintf(stderr, "Usage: CFA <testfile>\n");
		exit(1);
	}

	std::string filename(argv[1]);
	Transitive_closure(filename);
}

int main(int argc, char **argv)
{
	fprintf(stderr, "TEST START\n");
	runTest(argc, argv);
	fprintf(stderr, "TEST COMPLETE\n");
	return 0;
}
