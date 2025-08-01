#include <numeric>
#include <fstream>
#include <chrono>

#define INDEXTYPE uint32_t
#define VALUETYPE double

#define RHSDIM 16
#define ALIGN 32

#include "utility.h"
#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "Semirings.h"
#include "aligned.h"

using namespace std;

template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    for(auto& arr : vecofarr)
        arr.fill(static_cast<NT> (0));
}


template <typename NT, typename ALLOC, int DIM>
void VerifyMM (const vector< array<NT,DIM>, ALLOC > & control, const vector< array<NT,DIM>, ALLOC > & test)
{
    NT max_error = 0;
    pair<size_t, size_t> max_error_loc{0,0};

    for(size_t i=0; i<control.size(); ++i)
    {
        for(size_t j=0; j<DIM; ++j)
        {
            NT err = std::abs(control[i][j] - test[i][j]);
            if(err > max_error)
            {
                max_error = err;
                max_error_loc = make_pair(i, j);
            }
        }
    }

    cout << "Max error is: " << max_error 
         << " on y[" << max_error_loc.first << "][" << max_error_loc.second << "]=" 
         << test[max_error_loc.first][max_error_loc.second] << endl;

    NT machEps = machineEpsilon<NT>();
    cout << "Absolute machine epsilon is: " << machEps 
         << " and y[" << max_error_loc.first << "][" << max_error_loc.second 
         << "]*EPSILON becomes " << machEps * test[max_error_loc.first][max_error_loc.second] << endl;

    NT sqrtm = sqrt(static_cast<NT>(control.size()));
    cout << "sqrt(n) * relative error is: " << std::abs(machEps * test[max_error_loc.first][max_error_loc.second]) * sqrtm << endl;

    if ((std::abs(machEps * test[max_error_loc.first][max_error_loc.second]) * sqrtm) < std::abs(max_error))
        cout << "*** ATTENTION ***: error is more than sqrt(n) times the relative machine epsilon" << endl;
}


int main(int argc, char* argv[])
{
    bool syminput = false;
    bool binary = false;
    bool iscsc = false;
    INDEXTYPE m = 0, n = 0, nnz = 0, forcelogbeta = 0;
    string inputname;

    inputname = argv[1];
    string issym(argv[2]);
    if(issym == "sym") syminput = true;
    else if(issym == "nosym") syminput = false;
    else cout << "unrecognized option, assuming unsymmetric" << endl;

    string isbinary(argv[3]);
    if(isbinary == "text") binary = false;
    else if(isbinary == "binary") binary = true;
    else cout << "unrecognized option, assuming text file" << endl;

    if(argc > 4)
    {
        string type(argv[4]);
        if(type == "csc")
        {
            iscsc = true;
            cout << "Processing CSC binary" << endl;
        }
    }

    Csc<VALUETYPE, INDEXTYPE> * csc = nullptr;
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
    infile >> m >> n >> nnz;    // #{rows}-#{cols}-#{nonzeros}

    auto tstart = std::chrono::steady_clock::now();

    Triple<VALUETYPE, INDEXTYPE> * triples = new Triple<VALUETYPE, INDEXTYPE>[nnz];

    if (infile.is_open())
    {
        INDEXTYPE cnz = 0;    // current number of nonzeros
        while (! infile.eof() && cnz < nnz)
        {
            infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;    // row-col-value
            triples[cnz].row--;
            triples[cnz].col--;
            ++cnz;
        }
        assert(cnz == nnz);    
    }

    auto tend = std::chrono::steady_clock::now();
    cout << "Reading matrix in ascii took " 
            << std::chrono::duration<double>(tend - tstart).count() 
            << " seconds" << endl;

    cout << "converting to csc ... " << endl;
    csc= new Csc<VALUETYPE,INDEXTYPE>(triples, nnz, m, n);
    delete [] triples;

    BiCsb<VALUETYPE, INDEXTYPE> bicsb(*csc, 1, forcelogbeta);

    cout << "generating " << RHSDIM << " multi vectors... " << endl;
    typedef array<VALUETYPE, RHSDIM> PACKED;
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > x(n);
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_bicsb(m);
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_csc(m);

    fillzero<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_csc);
    fillzero<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_bicsb);
    for (size_t j = 0; j < n; ++j) {           // loop over rows of B (columns of A)
        for (size_t i = 0; i < RHSDIM; ++i) {  // loop over RHS (columns of B)
            x[j][i] = sin(i + j);
        }
    }


    typedef PTSRArray<VALUETYPE,VALUETYPE, RHSDIM> PTARR;
    cout << "starting SpMM ... " << endl;

    bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0]));
    csc_gaxpy_mm<RHSDIM>(*csc, &(x[0]), &(y_csc[0]));
    VerifyMM<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_csc, y_bicsb);

    delete csc;
}
