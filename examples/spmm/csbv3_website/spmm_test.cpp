#define NOMINMAX
#include <iostream> 
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <array>

#include "timer.gettimeofday.c"
#include "cilk_util.h"
#include "aligned.h"

#define INDEXTYPE uint32_t
#ifdef SINGLEPRECISION
	#define VALUETYPE float
#else
	#define VALUETYPE double
#endif

#ifndef RHSDIM
	#define RHSDIM 16
#endif
#define ALIGN 32

#include "utility.h"
#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"

using namespace std;


// notes: no binary files, currently does not deal with pattern files either. can edit
// bicsb is the final product for CSB. 
// bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0])); is the SpMM operation. Runs it once before running benchmarks
// dont bother with transpose version: bicsb_gespmvt, nor CSC for verification.
// RHSDIM = number of vectors
// dont know what RowImbalance and ColImbalance is and whether or not its important
// x(n) is our equivalent of B_val, instead of randomly generated we use sin(i+j);
// y_bicsb stores answer.

// need to fix their csc creation because no support for pattern or symmetric.
// they have symcsb but dont think it works well for our purposes

int main(int argc, char* argv[])
{
#ifndef CILK_STUB
	int gl_nworkers = __cilkrts_get_nworkers();
#endif
	INDEXTYPE m = 0, n = 0, nnz = 0, forcelogbeta = 0;
	string inputname;

	Csc<VALUETYPE, INDEXTYPE> * csc;
	// for no binary
	{
		
		cout << "reading input matrix in text(ascii)... " << endl;
		ifstream infile(inputname.c_str());
		char line[256];
		char c = infile.get();
		while(c == '%')
		{
			infile.getline(line,256);
			c = infile.get();
		}
		infile.unget();
		infile >> m >> n >> nnz;	// #{rows}-#{cols}-#{nonzeros}

		Triple<VALUETYPE, INDEXTYPE> * triples = new Triple<VALUETYPE, INDEXTYPE>[nnz];
	
		if (infile.is_open())
		{
			INDEXTYPE cnz = 0;	// current number of nonzeros
			while (! infile.eof() && cnz < nnz)
			{
				// no pattern .mtx files
				infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;	// row-col-value
				triples[cnz].row--;
				triples[cnz].col--;
				++cnz;
			}
			assert(cnz == nnz);	 // dont want this assertion. because it wont be correct.
		}

		cout << "converting to csc ... " << endl;
		// coo -> csc
		csc= new Csc<VALUETYPE,INDEXTYPE>(triples, nnz, m, n); // i think the issue is that 
		delete [] triples;
	}

	BiCsb<VALUETYPE, INDEXTYPE> bicsb(*csc, gl_nworkers, forcelogbeta);
		
	

	typedef array<VALUETYPE, RHSDIM> PACKED;
	vector< PACKED, aligned_allocator<PACKED, ALIGN> > x(n);
	vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_bicsb(m);

	fillzero<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_bicsb);

	typedef PTSRArray<VALUETYPE,VALUETYPE, RHSDIM> PTARR;		

	
	bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0])); // runs it one time to remove initial time


	bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0])); // 

}

