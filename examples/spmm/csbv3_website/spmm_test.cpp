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
#include <random>
#include <chrono>

#include "timer.gettimeofday.c"
// #include "cilk_util.h"
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
// #include "bmcsb.h"
#include "spvec.h"
#include "Semirings.h"

using namespace std;


template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    for(auto& arr : vecofarr)
        arr.fill(static_cast<NT> (0));
}

template <typename NT, typename ALLOC, int DIM>
void fillrandom (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    std::mt19937 engine{std::random_device{}()};
    std::uniform_real_distribution<NT> distribution(0.0f, 1.0f); 

    for(auto& arr : vecofarr)
        for(auto& val : arr)
            val = distribution(engine);
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

    if(argc < 2)
    {
        cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
        cout << "Assuming matrix.txt is the input, matrix is unsymmetric, and stored in text(ascii) file" << endl;
        inputname = "matrix.txt";
    }
    else if(argc < 3)
    {
        cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
        cout << "Assuming that the matrix is unsymmetric, and stored in text(ascii) file" << endl;
        inputname = argv[1];
    }
    else if(argc < 4)
    {
        cout << "Normal usage: ./a.out inputmatrix.mtx sym/nosym binary/text triples/csc" << endl;
        cout << "Assuming matrix is stored in text(ascii) file" << endl;
        inputname = argv[1];
        string issym(argv[2]);
        if(issym == "sym") syminput = true;
        else if(issym == "nosym") syminput = false;
        else cout << "unrecognized option, assuming nosym" << endl;
    }
    else
    {
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

        if(argc == 6)
            forcelogbeta = atoi(argv[5]);
    }

    Csc<VALUETYPE, INDEXTYPE> * csc = nullptr;
    if(binary)
    {
        FILE * f = fopen(inputname.c_str(), "r");
        if(!f)
        {
            cerr << "Problem reading binary input file\n";
            return 1;
        }
        if(iscsc)
        {
            fread(&n, sizeof(INDEXTYPE), 1, f);
            fread(&m, sizeof(INDEXTYPE), 1, f);
            fread(&nnz, sizeof(INDEXTYPE), 1, f);
        }
        else
        {
            fread(&m, sizeof(INDEXTYPE), 1, f);
            fread(&n, sizeof(INDEXTYPE), 1, f);
            fread(&nnz, sizeof(INDEXTYPE), 1, f);
        }
        if (m <= 0 || n <= 0 || nnz <= 0)
        {
            cerr << "Problem with matrix size in binary input file\n";    
            return 1;        
        }

        auto tstart = std::chrono::steady_clock::now();

        cout << "Reading matrix with dimensions: " << m << "-by-" << n << " having " << nnz << " nonzeros" << endl;

        INDEXTYPE * rowindices = new INDEXTYPE[nnz];
        VALUETYPE * vals = new VALUETYPE[nnz];
        INDEXTYPE * colindices = nullptr;
        INDEXTYPE * colpointers = nullptr;

        if(iscsc)
        {
            colpointers = new INDEXTYPE[n+1];
            size_t cols = fread(colpointers, sizeof(INDEXTYPE), n+1, f);
            if(cols != n+1)
            {
                cerr << "Problem with FREAD, aborting... " << endl;
                return -1;
            }
        }
        else
        {
            colindices = new INDEXTYPE[nnz];
            size_t cols = fread(colindices, sizeof(INDEXTYPE), nnz, f);
            if(cols != nnz)
            {
                cerr << "Problem with FREAD, aborting... " << endl;
                return -1;
            }
        }

        size_t rows = fread(rowindices, sizeof(INDEXTYPE), nnz, f);
        size_t nums = fread(vals, sizeof(VALUETYPE), nnz, f);

        if(rows != nnz || nums != nnz)
        {
            cerr << "Problem with FREAD, aborting... " << endl;
            return -1;
        }
        fclose(f);

        auto tend = std::chrono::steady_clock::now();
        cout << "Reading matrix in binary took " 
             << std::chrono::duration<double>(tend - tstart).count() 
             << " seconds" << endl;

        if(iscsc)
        {
            csc = new Csc<VALUETYPE, INDEXTYPE>();
            csc->SetPointers(colpointers, rowindices, vals , nnz, m, n, true);    // shallow copy
        }
        else
        {
            csc = new Csc<VALUETYPE, INDEXTYPE>(rowindices, colindices, vals , nnz, m, n);
            delete [] colindices;
            delete [] rowindices;
            delete [] vals;
        }
    }
    else
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
    }

    cout << "# workers: 1 (no Cilk)" << endl;

    BiCsb<VALUETYPE, INDEXTYPE> bicsb(*csc, 1, forcelogbeta);

    double mflops = (2.0 * static_cast<double>(nnz) * RHSDIM) / 1000000.0;
    cout << "generating " << RHSDIM << " multi vectors... " << endl;
    typedef array<VALUETYPE, RHSDIM> PACKED;
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > x(n);
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_bicsb(m);
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_csc(m);

    fillzero<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_csc);
    fillzero<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_bicsb);
    fillrandom<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(x);

    typedef PTSRArray<VALUETYPE,VALUETYPE, RHSDIM> PTARR;
    cout << "starting SpMV ... " << endl;
    cout << "Row imbalance is: " << RowImbalance(bicsb) << endl;
    cout << "Col imbalance is: " << ColImbalance(bicsb) << endl;

    timer_init();

    bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0]));
    auto t0 = std::chrono::steady_clock::now();

    for(int i=0; i < REPEAT; ++i)
    {
        bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0]));
    }
    auto t1 = std::chrono::steady_clock::now();

    double time = std::chrono::duration<double>(t1 - t0).count() / REPEAT;
    cout << "BiCSB time: " << time << " seconds" << endl;
    cout << "BiCSB mflop/sec: " << mflops / time << endl;

    // Verify with CSC (serial)
    csc_gaxpy_mm<RHSDIM>(*csc, &(x[0]), &(y_csc[0]));
    t0 = std::chrono::steady_clock::now();
    for(int i=0; i < REPEAT; ++i)
    {
        csc_gaxpy_mm<RHSDIM>(*csc, &(x[0]), &(y_csc[0]));
    }
    t1 = std::chrono::steady_clock::now();
    double csctime = std::chrono::duration<double>(t1 - t0).count() / REPEAT;
    cout << "CSC time: " << csctime << " seconds" << endl;
    cout << "CSC mflop/sec: " << mflops / csctime << endl;

    VerifyMM<VALUETYPE, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_csc, y_bicsb);

    delete csc;
}
