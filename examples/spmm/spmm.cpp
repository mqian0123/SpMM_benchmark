#include "../../src/benchmark.hpp"
#include <sys/stat.h>
#include <iostream>
#include <cstdint>

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
    int n = 15; // manually setting number of columns for very large matrices

    experiment_spmm_csr<double, int32_t>(params, n); 

    std::cout << "SpMM complete, checking correctness with Julia" << std::endl;
    std::string julia_cmd = "julia check_correctness.jl " + params.input + " " + params.output + " " + std::to_string(n);
    int ret = std::system(julia_cmd.c_str());

    if (ret != 0) {
        std::cerr << "Failed with code " << ret << std::endl;
        return ret;
    }
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

    // For COO format
    // bsp_array_t A_row = A_CSR.indices_0;
    // bsp_array_t A_col = A_CSR.indices_1;

    // Generate B_DMATR: NOTE: should not be double* but generic
    T* B_val = (T*) malloc(sizeof(T) * k * n);
    for (size_t i = 0; i < k; ++i) {
        for (size_t j = 0; j < n; ++j) {
            B_val[i*n + j] = sin(i + j);  // random example data
        }
    }

    // Allocate result: C = m x n
    T* C_val = (T*) calloc(m * n, sizeof(T));

    auto time = benchmark(
    []() { // setup is nothing right now
    },
        // this is the run() command
        [&C_val, &A_ptr, &A_val, &A_idx, &B_val, &nnz, &n, &m]() {
            I col_a, p, pmax;
            T val_a;

            memset(C_val, 0, sizeof(T) * m * n); // reset C_val
            
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
        }
    );

    // NOTE: COOR FORMAT
    // auto time = benchmark(
    // []() { // setup is nothing right now
    // },
    //     // this is the run() command
    //     [&C_val, &A_row, &A_val, &A_col, &B_val, &nnz, &n, &m]() {
    //         I row, col;
    //         T val;

    //         memset(C_val, 0, sizeof(T) * m * n); // reset C_val

    //         for(int i=0; i<nnz; ++i) {
    //             bsp_array_read(A_row, i, row);
    //             bsp_array_read(A_col, i, col);
    //             bsp_array_read(A_val, i, val);

    //             for(int j=0; j<n; ++j) {
    //                 C_val[row*n + j] += val * B_val[col*n + j];
    //             }
    //         }
    //     }
    // );

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
    measurements["time"] = time;
    measurements["memory"] = 0; // not implemented yet in the other implementations as well
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}