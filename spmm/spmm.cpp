#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>

#include "mkl_spblas.h" // Intel MKL
// #include <ittnotify.h>  // Intel Advisor

#define ALIGN 64
#define RHSDIM 4   // number of columns in B, used for CSB

#include "csb_library/csb.h"    // CSB Implementation

// static __itt_domain* domain = __itt_domain_create("SpMM");
// static __itt_string_handle* task_name = __itt_string_handle_create("test");

// ================== Aligned allocator ==================
template <class T, std::size_t Alignment>
struct AlignedAllocator {
    static_assert(Alignment && ((Alignment & (Alignment - 1)) == 0), "Alignment must be a power of two");
    using value_type = T;
    AlignedAllocator() noexcept {}
    template <class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    T* allocate(std::size_t n) {
        void* p = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Alignment)); }
    template <class U> struct rebind { using other = AlignedAllocator<U, Alignment>; };
};
template <class T, class U, std::size_t A>
constexpr bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return true; }
template <class T, class U, std::size_t A>
constexpr bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) noexcept { return false; }

// ================== CSR container ==================
template <typename Index = int32_t, typename Value = double, std::size_t Alignment = 64>
struct CSR {
    using index_t = Index;
    using value_t = Value;
    Index nrows = 0, ncols = 0;
    std::vector<index_t, AlignedAllocator<index_t, Alignment>> row_ptr;
    std::vector<index_t, AlignedAllocator<index_t, Alignment>> col_idx;
    std::vector<value_t, AlignedAllocator<value_t, Alignment>> values;
};

// From CSB Library
template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    for(auto& arr : vecofarr)
        arr.fill(static_cast<NT> (0));
}

// ================== Loader ==================
template <typename Index = int32_t, typename Value = double, std::size_t Alignment = 64>
CSR<Index, Value, Alignment> load_mtx(const std::string& filename) {

    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(fin, line);
    std::istringstream header(line);
    std::string banner, mtx, format, field, symmetry;
    header >> banner >> mtx >> format >> field >> symmetry;
    
    if (banner != "%%MatrixMarket" || mtx != "matrix" || format != "coordinate") {
        throw std::runtime_error("Only MatrixMarket 'matrix coordinate' is supported");
    }

    bool is_pattern   = (field == "pattern");
    bool is_numeric   = (field == "real" || field == "integer");
    bool is_symmetric = (symmetry == "symmetric");
    
    if (!(is_pattern || is_numeric)) {
        throw std::runtime_error("Unsupported field: " + field);
    }

    // Skip comments
    while (std::getline(fin, line)) {
        if (line[0] != '%') {
            break;
        }
    }

    // Get Dimensions and NNZ
    Index nrows=0, ncols=0;
    long long nnz;
    std::istringstream ss(line);
    if (!(ss >> nrows >> ncols >> nnz)) {
        throw std::runtime_error("Failed to parse size line");
    }
    if (nrows <= 0 || ncols <= 0 || nnz < 0) {
        throw std::runtime_error("Invalid matrix dimensions/nnz");
    }

    struct Triplet { Index r, c; Value v; };
    std::vector<Triplet> coo;
    coo.reserve(is_symmetric ? static_cast<size_t>(nnz)*2ull : static_cast<size_t>(nnz));

    // Parse values
    long long read_entries = 0;
    while (read_entries < nnz && std::getline(fin, line)) {
        bool only_ws = true;
        for (char ch : line) { if (!std::isspace(static_cast<unsigned char>(ch))) { only_ws = false; break; } }
        if (only_ws) continue;
        if (!line.empty() && line[0] == '%') continue;

        Index row_idx, col_idx;
        Value value = static_cast<Value>(1.0); // Default 1.0 for type == pattern (no values)
        std::istringstream ss(line);
        if (!(ss >> row_idx >> col_idx)) {
            throw std::runtime_error("Failed to parse entry indices");
        }
        if (is_numeric && !(ss >> value)) {
            throw std::runtime_error("Failed to parse entry value");
        }
        
        // Switch to zero-based indexing
        row_idx--;
        col_idx--;

        coo.push_back({row_idx, col_idx, value}); 
        if (is_symmetric && row_idx != col_idx) {
            coo.push_back({col_idx, row_idx, value}); // add corresponding coordinate if symmetric
        }
        ++read_entries;
    }
    if (read_entries != nnz) {
        throw std::runtime_error("Entry count mismatch");
    } 

    // Result is sorted CSR
    std::sort(coo.begin(), coo.end(), [](const Triplet& a, const Triplet& b){
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    // Sums up duplicate coordinates
    std::vector<Triplet> merged;
    merged.reserve(coo.size());
    for (size_t i = 0; i < coo.size();) {
        Index r = coo[i].r, c = coo[i].c;
        Value sum = 0;
        size_t j = i;
        while (j < coo.size() && coo[j].r == r && coo[j].c == c) {
            sum += coo[j].v;
            ++j;
        }
        merged.push_back({r, c, sum});
        i = j;
    }
    coo.swap(merged);

    // Set up CSR
    CSR<Index, Value, Alignment> csr;
    csr.nrows = nrows;
    csr.ncols = ncols;
    csr.row_ptr.assign(static_cast<size_t>(nrows)+1, 0);
    csr.col_idx.resize(coo.size());
    csr.values.resize(coo.size());

    // Populate CSR

    // Counts number of values at each row
    for (const auto& t : coo) {
        csr.row_ptr[t.r + 1]++;
    }

    // Updates row_ptr to be cumulative
    for (size_t i = 0; i < static_cast<size_t>(nrows); ++i) {
        csr.row_ptr[i+1] += csr.row_ptr[i];
    }

    // Fills in col_idx and values
    std::vector<Index, AlignedAllocator<Index, Alignment>> cursor = csr.row_ptr;
    for (const auto& t : coo) {
        Index i = cursor[t.r]++;
        csr.col_idx[i] = t.c;
        csr.values[i] = t.v;
    }

    return csr;
}


// ================== CSR ==================
std::vector<double> experiment_spmm_csr(
    size_t n, size_t m,
    const int32_t* A_ptr, const int32_t* A_idx, const double* A_val,
    const double* B_val
) {
    std::vector<double> C_val(m * n, 0.0);
    

    std::cout << "Beginning SpMM CSR" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Advisor
    // __itt_resume();
    // __itt_task_begin(domain, __itt_null, __itt_null, task_name);

    #pragma omp parallel for schedule(dynamic)
    for (size_t row = 0; row < m; ++row) {
        int32_t row_start = A_ptr[row];
        int32_t row_end   = A_ptr[row + 1];
        for (int32_t idx = row_start; idx < row_end; ++idx) {
            int32_t col = A_idx[idx];
            double val = A_val[idx];
            double* c_row = &C_val[row * n];
            const double* b_row = &B_val[col * n];
            #pragma omp simd aligned(c_row,b_row:32)
            for (size_t j = 0; j < n; ++j) c_row[j] += val * b_row[j];
        }
    }

    // __itt_task_end(domain);
    // __itt_pause();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_csr time: " << std::fixed << std::setprecision(6) 
              << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";
    
    return C_val;
}

// ================== MKL ==================
std::vector<double> experiment_spmm_mkl(
    size_t n, size_t m, size_t k,
    int32_t* A_ptr, int32_t* A_idx, double* A_val, double* B_val
) {
    sparse_matrix_t A;
    mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, m, k, A_ptr, A_ptr+1, A_idx, A_val);
    struct matrix_descr descr; descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_sparse_optimize(A);
    std::vector<double> C_val(m * n, 0.0);
    double alpha=1.0, beta=0.0;


    std::cout << "Beginning SpMM MKL" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Advisor
    // __itt_resume();
    // __itt_task_begin(domain, __itt_null, __itt_null, task_name);

    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, B_val, n, n, beta, C_val.data(), n);
    
    // __itt_task_end(domain);
    // __itt_pause();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_mkl time: " << std::fixed << std::setprecision(6)
              << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";


    return C_val;
}

// ================== CSB ==================
template <typename Index = int32_t, typename Value = double, std::size_t Alignment = 32>
std::vector<double> experiment_spmm_csb(
    size_t n,                                  // #columns in B
    const CSR<Index, Value, Alignment>& A_csr // pass CSR with template args
) {
    // Convert CSR to triples for BiCsb
    std::vector<Triple<Value,Index>> triples;
    triples.reserve(A_csr.col_idx.size());

    for (Index r = 0; r < A_csr.nrows; ++r) {
        Index row_start = A_csr.row_ptr[r];
        Index row_end   = A_csr.row_ptr[r+1];
        for (Index idx = row_start; idx < row_end; ++idx) {
            triples.push_back({r, A_csr.col_idx[idx], A_csr.values[idx]});
        }
    }

    Csc<Value,Index> csc(triples.data(), triples.size(), A_csr.nrows, A_csr.ncols);
    int32_t forcelogbeta = 0;
    BiCsb<Value,Index> bicsb(csc, 1, forcelogbeta);

    typedef array<Value, RHSDIM> PACKED;
    std::vector<PACKED, aligned_allocator<PACKED, Alignment>> x(A_csr.ncols);
    std::vector<PACKED, aligned_allocator<PACKED, Alignment>> y_bicsb(A_csr.nrows);
    fillzero<Value, aligned_allocator<PACKED, Alignment>, RHSDIM>(y_bicsb);

    for (size_t i = 0; i < A_csr.ncols; ++i) {
        for (size_t j = 0; j < n; ++j) {
            x[i][j] = sin(i + j);
        }
    }

    typedef PTSRArray<Value,Value,RHSDIM> PTARR;
    
    bicsb_gespmv<PTARR>(bicsb, &x[0], &y_bicsb[0]);
    fillzero<Value, aligned_allocator<PACKED, Alignment>, RHSDIM>(y_bicsb);


    std::cout << "Beginning SpMM CSB" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Advisor
    // __itt_resume();
    // __itt_task_begin(domain, __itt_null, __itt_null, task_name);

    bicsb_gespmv<PTARR>(bicsb, &x[0], &y_bicsb[0]);

    // __itt_task_end(domain);
    // __itt_pause();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_csb time: " << std::fixed << std::setprecision(6)
              << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";


    std::vector<Value> C_val_bicsb;
    C_val_bicsb.reserve(A_csr.nrows * n);

    for (Index row = 0; row < A_csr.nrows; ++row) {
        for (size_t col = 0; col < n; ++col) {
            C_val_bicsb.push_back(y_bicsb[row][col]);
        }
    }

    return C_val_bicsb;
}
// ================== Compare ==================
bool compare_dense_matrices(const std::vector<double>& C1, const std::vector<double>& C2,
                            size_t rows, size_t cols, double tol=1e-12)
{
    for (size_t i = 0; i < rows*cols; ++i) {
        if (std::abs(C1[i]-C2[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": " << C1[i] << " vs " << C2[i] << "\n";
            return false;
        }
    }
    return true;
}

// ================== Main ==================
int main(int argc, char** argv) {
    // __itt_pause();

    if (argc < 3) { 
        std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx> <n_columns_B>" << std::endl; 
        return 1; 
    }
    std::string matrix_file = argv[1];
    int n = std::atoi(argv[2]);
    if (n <= 0) { 
        std::cerr << "Error: columns > 0" << std::endl; 
        return 1; 
    }

    auto A_csr = load_mtx<int32_t,double,32>(matrix_file);
    size_t k = A_csr.ncols;
    size_t m = A_csr.nrows;

    std::vector<double, AlignedAllocator<double,32>> B_val(k*n);
    
    // Generating dense matrix B, could change from sin
    for (size_t i=0;i<k;i++) {
        for (int j=0;j<n;j++) {
            B_val[i*n+j] = sin(i+j); 
        }
    }
    
    auto C_csr = experiment_spmm_csr(n, m, 
        A_csr.row_ptr.data(), A_csr.col_idx.data(), A_csr.values.data(), B_val.data());

    auto C_mkl = experiment_spmm_mkl(n,m,k,
        A_csr.row_ptr.data(), A_csr.col_idx.data(), A_csr.values.data(), B_val.data());

    auto C_csb = experiment_spmm_csb(n, A_csr);

    std::cout << "CSR vs MKL match: " << (compare_dense_matrices(C_csr,C_mkl,m,n) ? "YES" : "NO") << "\n";
    std::cout << "CSR vs CSB match: " << (compare_dense_matrices(C_csr,C_csb,m,n) ? "YES" : "NO") << "\n";
    std::cout << "MKL vs CSB match: " << (compare_dense_matrices(C_mkl,C_csb,m,n) ? "YES" : "NO") << "\n";

    return 0;
}
