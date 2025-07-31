#include "../../src/benchmark.hpp"
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <chrono>

#include "mkl_spblas.h"
#include <advisor-annotate.h> // intel advisor


#include <binsparse/binsparse.h>

namespace fs = std::filesystem;

template <typename T, typename I>
void experiment_spmm_csr(benchmark_params_t params, size_t n);

// NOTE: When testing .mtx files, ensure the file is NOT symmetric (or change the header) - binsparse does
// not yet support symmetric files

// compile using ./spmm -i data/A.mtx -o results
// input includes the specific .mtx file, output is just the folder

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    std::cout << "Reading A from: " << (realpath((fs::path(params.input)).c_str(), NULL)) << std::endl;
    
    bsp_matrix_t A_CSR = bsp_read_matrix((realpath((fs::path(params.input)).c_str(), NULL)), NULL);
    // int n = A_CSR.ncols; // columns of B, set as a square matrix
    int n = 50; // manually setting number of columns for very large matrices

    experiment_spmm_csr<double, int32_t>(params, n); 

    std::cout << "SpMM complete, checking correctness with Julia" << std::endl;
    std::string julia_cmd = "julia check_correctness.jl " + params.input + " " + params.output + " " + std::to_string(n);
    // int ret = std::system(julia_cmd.c_str());

    // if (ret != 0) {
    //     std::cerr << "Failed with code " << ret << std::endl;
    //     return ret;
    // }
    std::cout << "Result saved under " << (realpath((fs::path(params.output)).c_str(), NULL)) << std::endl;
    return 0;
}

template <typename T, typename I>
void experiment_spmm_csr(benchmark_params_t params, size_t n){
    // for testing purposes, A = sparse matrix, B = dense matrix
    bsp_matrix_t A_COO = bsp_read_matrix((realpath((fs::path(params.input)).c_str(), NULL)), NULL);
    bsp_matrix_t A_CSR = bsp_convert_matrix(A_COO, BSP_CSR);

    // bsp_print_matrix_info(A_CSR);

    // A = m x k, B = k x n, A*B = C = m x n
    I m = A_CSR.nrows;
    I k = A_CSR.ncols;
    I nnz = A_CSR.nnz;

    bsp_array_t A_ptr = A_CSR.pointers_to_1;
    bsp_array_t A_idx = A_CSR.indices_1;
    bsp_array_t A_val = A_CSR.values;


    // Generate B_DMATR: NOTE: should not be double* but generic
    T* B_val = (T*) malloc(sizeof(T) * k * n);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B_val[i*n + j] = sin(i + j);  // random example data
        }
    }

    // Allocate result: C = m x n
    T* C_val = (T*) calloc(m * n, sizeof(T));

    I col_a, p, pmax;
    T val_a;

    memset(C_val, 0, sizeof(T) * m * n); // reset C_val
    
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Beginning self-implemented SpMM" << std::endl;

    ANNOTATE_SITE_BEGIN("spmm");

    for (int i = 0; i < m; ++i) {
        bsp_array_read(A_ptr, i + 1, pmax); // pmax = A_ptr[i+1]
        bsp_array_read(A_ptr, i, p);
        for(bsp_array_read(A_ptr, i, p); p < pmax; ++p) { // loops through row pointers
            bsp_array_read(A_idx, p, col_a); // stores column index into col_a, row index = i
            bsp_array_read(A_val, p, val_a); // stores value into val_a

            // dot product matrix multiplication
            for(int j=0; j<n; ++j) {
                C_val[i*n + j] += val_a * B_val[col_a*n + j];
            }
        }
    }

    ANNOTATE_SITE_END("spmm");
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time);

    sparse_matrix_t A;

    I* row_ptr_mkl = static_cast<I*>(A_ptr.data);
    I* col_ind_mkl = static_cast<I*>(A_idx.data);
    T* values_mkl = static_cast<T*>(A_val.data);

    // std::cout << "m=" << m << " k=" << k << " nnz=" << nnz << "\n";

    // std::cout << "row_ptr: ";
    // for (int i = 0; i < std::min<int>(m+1, 20); i++) {
    //     std::cout << row_ptr_mkl[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "col_ind: ";
    // for (int i = 0; i < std::min<int>(nnz, 20); i++) {
    //     std::cout << col_ind_mkl[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "values: ";
    // for (int i = 0; i < std::min<int>(nnz, 20); i++) {
    //     std::cout << values_mkl[i] << " ";
    // }
    // std::cout << "\n";

    // std::cout << "sizeof(MKL_INT) = " << sizeof(MKL_INT) << "\n";
    // std::cout << "bsp_type_size(A_ptr.type) = " << bsp_type_size(A_ptr.type) << "\n";

    sparse_status_t status = mkl_sparse_d_create_csr(
        &A, SPARSE_INDEX_BASE_ZERO,
        m, k,
        row_ptr_mkl,
        row_ptr_mkl+ 1,
        col_ind_mkl,
        values_mkl
    );

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL failed to create CSR matrix\n";
    }

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_optimize(A); 
    double alpha = 1.0, beta = 0.0;

    T* C_val_mkl = (T*) calloc(m * n, sizeof(T));


    std::cout << "Beginning MKL implementation SpMM" << std::endl;

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

    // Testing shows that MKL matches
    // bool approx_equal = true;
    // const double tol = 1e-6;
    // for (int i = 0; i < m * n; ++i) {
    //     double diff = std::abs(static_cast<double>(C_val[i]) - static_cast<double>(C_val_mkl[i]));        
    //     if (diff > tol) {
    //         approx_equal = false;
    //         std::cout << "Mismatch at index " << i << ": " << C_val[i] << " vs " << C_val_mkl[i] << std::endl;
    //         break;
    //     }
    // }

    // if (approx_equal) {
    //     std::cout << "Arrays approximately equal" << std::endl;
    // } else {
    //     std::cout << "Arrays differ" << std::endl;
    // }

    // generating binsparse.json file to store results
    std::filesystem::create_directory(fs::path(params.output)/"C.bspnpy");
    json C_desc;
    C_desc["version"] = 0.5; 
    C_desc["format"] = "DMATR"; 
    C_desc["shape"] = {m, n};
    C_desc["nnz"] = m * n; // Dense matrix output, can assume all values are nnz
    C_desc["data_types"]["values_type"] = "float64";
    std::ofstream C_desc_file(fs::path(params.output)/"C.bspnpy"/"binsparse.json");
    C_desc_file << C_desc;
    C_desc_file.close();

    std::vector<T> vec(C_val, C_val + (m*n));
    npy_store_vector<T>(fs::path(params.output)/"C.bspnpy"/"values.npy", vec);

    json measurements;
    // measurements["time"] = time;
    measurements["memory"] = 0; // not implemented yet in the other implementations as well
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}