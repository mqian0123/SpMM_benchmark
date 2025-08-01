#include "../../src/benchmark.hpp"
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <numeric>
#include <fstream>
#include <chrono>
#include <cmath>

#define RHSDIM 16
#define ALIGN 32

#include "utility.h"
#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "Semirings.h"
#include "aligned.h"
#include "mkl_spblas.h"
#include <advisor-annotate.h> // intel advisor


#include <binsparse/binsparse.h>

namespace fs = std::filesystem;

template <typename T, typename I>
T* experiment_spmm_csr(benchmark_params_t params, size_t n, I m, I k, I nnz, I* A_ptr, I* A_idx, T* A_val, T* B_val);

template <typename T, typename I>
T* experiment_spmm_mkl(benchmark_params_t params, size_t n, I m, I k, I nnz, I* A_ptr, I* A_idx, T* A_val, T* B_val);

template <typename T, typename I>
T* experiment_spmm_csb(benchmark_params_t params, size_t n, I m, I k, I nnz);

template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr);

template <typename T>
bool compare_dense_matrices(const T* C1, const T* C2, size_t rows, size_t cols, T tol = 1e-6);


// NOTE: When testing .mtx files, ensure the file is NOT symmetric (or change the header) - binsparse does
// not yet support symmetric files

// compile using ./spmm -i data/A.mtx -o results
// input includes the specific .mtx file, output is just the folder

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    size_t n = RHSDIM;
    std::cout << "Reading A from: " << (realpath((fs::path(params.input)).c_str(), NULL)) << std::endl;
    std::cout << "Number of vectors: " << n << "\n\n";
    bsp_matrix_t A_COO = bsp_read_matrix((realpath((fs::path(params.input)).c_str(), NULL)), NULL);
    bsp_matrix_t A_CSR = bsp_convert_matrix(A_COO, BSP_CSR);

    // A = m x k, B = k x n, C = m x n
    int32_t m = A_CSR.nrows;
    int32_t k = A_CSR.ncols;
    int32_t nnz = A_CSR.nnz;

    bsp_array_t A_ptr_bsp = A_CSR.pointers_to_1;
    bsp_array_t A_idx_bsp = A_CSR.indices_1;
    bsp_array_t A_val_bsp = A_CSR.values;

    int32_t* A_ptr = static_cast<int32_t*>(A_ptr_bsp.data);
    int32_t* A_idx = static_cast<int32_t*>(A_idx_bsp.data);
    double* A_val = static_cast<double*>(A_val_bsp.data);

    // Generate B_DMATR: NOTE: should not be double* but generic
    double* B_val = (double*) malloc(sizeof(double) * k * n);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B_val[i*n + j] = sin(i + j);  // random example data
        }
    }

    double* result_mkl = experiment_spmm_mkl<double, int32_t>(params, n, m, k, nnz, A_ptr, A_idx, A_val, B_val); 
    double* result_csr = experiment_spmm_csr<double, int32_t>(params, n, m, k, nnz, A_ptr, A_idx, A_val, B_val); 
    double* result_csb = experiment_spmm_csb<double, int32_t>(params, n, m, k, nnz);

    std::cout << "SpMM complete, checking correctness" << std::endl;
    // std::string julia_cmd = "julia check_correctness.jl " + params.input + " " + params.output + " " + std::to_string(n);
    // int ret = std::system(julia_cmd.c_str());

    // if (ret != 0) {
    //     std::cerr << "Failed with code " << ret << std::endl;
    //     return ret;
    // }
    if(compare_dense_matrices(result_csr, result_mkl, m, n) 
    && compare_dense_matrices(result_csr, result_csb, m, n)
    && compare_dense_matrices(result_csb, result_mkl, m, n))
        std::cout << "All implementations within tolerance of each other" << std::endl;
    else
        std::cout << "Mismatch between the different implementations" << std::endl;
    
    return 0;
}

template <typename T, typename I>
T* experiment_spmm_csr(benchmark_params_t params, size_t n, I m, I k, I nnz, 
    I* A_ptr, I* A_idx, T* A_val, T* B_val){

    T* C_val = (T*) calloc(m * n, sizeof(T));  // Allocate result: C = m x n
    // memset(C_val, 0, sizeof(T) * m * n); // reset C_val for multiple trials

    I col_a, p, pmax;
    T val_a;

    std::cout << "Beginning SpMM CSR" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    ANNOTATE_SITE_BEGIN("spmm");

    #pragma omp parallel for schedule(dynamic)
    for (I row = 0; row < m; ++row) {
        I row_start = A_ptr[row];
        I row_end = A_ptr[row + 1];
        for (I idx = row_start; idx < row_end; ++idx) {
            I col = A_idx[idx];
            T val = A_val[idx];

            for (I j = 0; j < n; ++j) {
                C_val[row * n + j] += val * B_val[col * n + j];
            }
        }
    }

    ANNOTATE_SITE_END("spmm");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_csr time: " << std::fixed << std::setprecision(6) << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";

    return C_val;
}

template <typename T, typename I>
T* experiment_spmm_mkl(benchmark_params_t params, size_t n, I m, I k, I nnz, 
    I* A_ptr, I* A_idx, T* A_val, T* B_val) {

    sparse_matrix_t A;

    sparse_status_t status = mkl_sparse_d_create_csr(
        &A, SPARSE_INDEX_BASE_ZERO,
        m, k,
        A_ptr,
        A_ptr+ 1,
        A_idx,
        A_val
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL failed to create CSR matrix\n";
    }

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_optimize(A); 
    double alpha = 1.0, beta = 0.0;

    T* C_val_mkl = (T*) calloc(m * n, sizeof(T));

    std::cout << "Beginning SpMM MKL" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    ANNOTATE_SITE_BEGIN("mkl_spmm");
    // C = A * B, set alpha=1, beta=0 -> no accumulation
    status = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr,
                SPARSE_LAYOUT_ROW_MAJOR,
                    B_val, n,
                    n,               // LDB = num columns of B
                    beta,
                    C_val_mkl, n     // LDC = num columns of C
                );

    ANNOTATE_SITE_END("mkl_spmm");
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_mkl time: " << std::fixed << std::setprecision(6) << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";

    return C_val_mkl;
}

template <typename T, typename I>
T* experiment_spmm_csb(benchmark_params_t params, size_t n, I m, I k, I nnz) {
    
    Csc<T, I> * csc = nullptr;
    ifstream infile(fs::absolute(fs::path(params.input)).c_str());
    char line[256];

    while (infile.peek() == '%') infile.ignore(4096, '\n');
    infile.ignore(4096, '\n'); 

    auto tstart = std::chrono::steady_clock::now();

    Triple<T, I> * triples = new Triple<T, I>[nnz];

    if (infile.is_open())
    {
        I cnz = 0;    // current number of nonzeros
        while (! infile.eof() && cnz < nnz)
        {
            infile >> triples[cnz].row >> triples[cnz].col >> triples[cnz].val;    // row-col-value
            triples[cnz].row--;
            triples[cnz].col--;
            ++cnz;
        }
        assert(cnz == nnz);    
    }

    csc= new Csc<T,I>(triples, nnz, m, k);
    delete [] triples;

    I forcelogbeta = 0;
    BiCsb<T, I> bicsb(*csc, 1, forcelogbeta);

    typedef array<T, RHSDIM> PACKED;
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > x(k);
    vector< PACKED, aligned_allocator<PACKED, ALIGN> > y_bicsb(m);
 
    fillzero<T, aligned_allocator<PACKED, ALIGN>, RHSDIM>(y_bicsb);
    for (size_t i = 0; i < k; ++i) {           // loop over rows of B (columns of A)
        for (size_t j = 0; j < n; ++j) {  // loop over RHS (columns of B)
            x[i][j] = sin(i + j);
        }
    }

    typedef PTSRArray<T,T, RHSDIM> PTARR;
    // cout << "starting SpMM ... " << endl;
    
    std::cout << "Beginning SpMM CSB" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    bicsb_gespmv<PTARR>(bicsb, &(x[0]), &(y_bicsb[0]));
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_mkl time: " << std::fixed << std::setprecision(6) << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";

    delete csc;

    T* C_val_bicsb = (T*) calloc(m * n, sizeof(T));

    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            C_val_bicsb[row * n + col] = y_bicsb[row][col];
        }
    }

    return C_val_bicsb;
}

template <typename T>
bool compare_dense_matrices(const T* C1, const T* C2, size_t rows, size_t cols, T tol) 
{
    for (size_t i = 0; i < rows * cols; ++i) {
        T diff = std::abs(C1[i] - C2[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at index " << i << ": "
                      << C1[i] << " vs " << C2[i] 
                      << " (diff = " << diff << ")" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    for(auto& arr : vecofarr)
        arr.fill(static_cast<NT> (0));
}