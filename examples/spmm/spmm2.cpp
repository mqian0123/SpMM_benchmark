#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <limits>
#include <new>
#include <sys/stat.h>
#include <cstdint>
#include <numeric>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>

#include "mkl_spblas.h"
#include <ittnotify.h>

#define ALIGN 32
#define RHSDIM 16

#include "utility.h"
#include "triple.h"
#include "csc.h"
#include "bicsb.h"
#include "Semirings.h"
#include "aligned.h"

static __itt_domain* domain = __itt_domain_create("SpMM");
static __itt_string_handle* task_name = __itt_string_handle_create("test");
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
template <typename Index = int32_t, typename Value = double, std::size_t Alignment = 32>
struct CSR {
    using index_t = Index;
    using value_t = Value;
    Index nrows = 0, ncols = 0;
    std::vector<index_t, AlignedAllocator<index_t, Alignment>> row_ptr;
    std::vector<index_t, AlignedAllocator<index_t, Alignment>> col_idx;
    std::vector<value_t, AlignedAllocator<value_t, Alignment>> values;
};

template <typename NT, typename ALLOC, int DIM>
void fillzero (vector< array<NT,DIM>, ALLOC > & vecofarr)
{
    for(auto& arr : vecofarr)
        arr.fill(static_cast<NT> (0));
}
// ================== CSR Loader (MatrixMarket) ==================
template <typename Index = int32_t, typename Value = double, std::size_t Alignment = 32>
CSR<Index, Value, Alignment> load_mtx_to_csr_canonical(const std::string& filename) {
    auto fail = [&](const std::string& msg){ throw std::runtime_error(msg); };

    std::ifstream fin(filename);
    if (!fin) fail("Could not open file: " + filename);

    std::string line;
    if (!std::getline(fin, line)) fail("Empty file");
    std::istringstream hdr(line);
    std::string banner, mtx, format, field, symmetry;
    hdr >> banner >> mtx >> format >> field >> symmetry;
    if (banner != "%%MatrixMarket" || mtx != "matrix" || format != "coordinate")
        fail("Only MatrixMarket 'matrix coordinate' is supported");

    bool is_pattern   = (field == "pattern");
    bool is_numeric   = (field == "real" || field == "integer");
    bool is_symmetric = (symmetry == "symmetric");
    if (!(is_pattern || is_numeric)) fail("Unsupported field: " + field);
    if (!(symmetry == "general" || symmetry == "symmetric")) fail("Unsupported symmetry: " + symmetry);

    auto is_comment_or_blank = [](const std::string& s){
        for (char ch : s) { if (!std::isspace(static_cast<unsigned char>(ch))) return ch == '%'; } 
        return true;
    };

    while (std::getline(fin, line)) if (!is_comment_or_blank(line)) break;
    if (!fin) fail("Missing size line");

    long long nrows_ll=0, ncols_ll=0, nnz_decl_ll=0;
    std::istringstream ss(line);
    if (!(ss >> nrows_ll >> ncols_ll >> nnz_decl_ll)) fail("Failed to parse size line");
    if (nrows_ll <= 0 || ncols_ll <= 0 || nnz_decl_ll < 0) fail("Invalid matrix dimensions/nnz");

    auto safe_cast_idx = [&](long long x)->Index{
        if (x < 0 || x > static_cast<long long>(std::numeric_limits<Index>::max()))
            fail("Index type overflow");
        return static_cast<Index>(x);
    };
    Index nrows = safe_cast_idx(nrows_ll);
    Index ncols = safe_cast_idx(ncols_ll);
    long long nnz_decl = nnz_decl_ll;

    struct Triplet { Index r, c; Value v; };
    std::vector<Triplet> coo;
    coo.reserve(is_symmetric ? static_cast<size_t>(nnz_decl)*2ull : static_cast<size_t>(nnz_decl));

    auto push_entry = [&](Index r, Index c, Value v){
        if (r < 0 || r >= nrows || c < 0 || c >= ncols) fail("Entry index out of bounds");
        coo.push_back({r, c, v});
    };

    long long read_entries = 0;
    while (read_entries < nnz_decl && std::getline(fin, line)) {
        bool only_ws = true;
        for (char ch : line) { if (!std::isspace(static_cast<unsigned char>(ch))) { only_ws = false; break; } }
        if (only_ws) continue;
        if (!line.empty() && line[0] == '%') continue;

        long long ri1, ci1;
        Value v = static_cast<Value>(1.0);
        std::istringstream ss(line);
        if (!(ss >> ri1 >> ci1)) fail("Failed to parse entry indices");
        if (is_numeric && !(ss >> v)) fail("Failed to parse entry value");

        Index ri = safe_cast_idx(ri1 - 1);
        Index ci = safe_cast_idx(ci1 - 1);
        push_entry(ri, ci, v);
        if (is_symmetric && ri != ci) push_entry(ci, ri, v);
        ++read_entries;
    }
    if (read_entries != nnz_decl) fail("Entry count mismatch");

    std::sort(coo.begin(), coo.end(), [](const Triplet& a, const Triplet& b){
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

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

    CSR<Index, Value, Alignment> csr;
    csr.nrows = nrows;
    csr.ncols = ncols;
    csr.row_ptr.assign(static_cast<size_t>(nrows)+1, 0);
    csr.col_idx.resize(coo.size());
    csr.values.resize(coo.size());

    for (const auto& t : coo) csr.row_ptr[t.r + 1]++;
    for (size_t i = 0; i < static_cast<size_t>(nrows); ++i) csr.row_ptr[i+1] += csr.row_ptr[i];

    std::vector<Index, AlignedAllocator<Index, Alignment>> cursor = csr.row_ptr;
    for (const auto& t : coo) {
        Index dst = cursor[t.r]++;
        csr.col_idx[dst] = t.c;
        csr.values[dst] = t.v;
    }

    return csr;
}


// ================== CSR SIMD SpMM ==================
std::vector<double> experiment_spmm_csr_simd(
    size_t n, size_t m,
    const int32_t* A_ptr, const int32_t* A_idx, const double* A_val,
    const double* B_val
) {
    std::vector<double> C_val(m * n, 0.0);
    __itt_resume();
    __itt_task_begin(domain, __itt_null, __itt_null, task_name);

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
    __itt_task_end(domain);
    __itt_pause();
    return C_val;
}

// ================== MKL SpMM ==================
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
    __itt_resume();
    __itt_task_begin(domain, __itt_null, __itt_null, task_name);
    mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, alpha, A, descr,
                    SPARSE_LAYOUT_ROW_MAJOR, B_val, n, n, beta, C_val.data(), n);
    __itt_task_end(domain);
    __itt_pause();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_mkl time: "
              << std::fixed << std::setprecision(6)
              << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";

    return C_val;
}

// CSB SpMM
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
    __itt_resume();
    __itt_task_begin(domain, __itt_null, __itt_null, task_name);
    bicsb_gespmv<PTARR>(bicsb, &x[0], &y_bicsb[0]);
    __itt_task_end(domain);
    __itt_pause();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_csb time: "
              << std::fixed << std::setprecision(6)
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
    __itt_pause();

    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <matrix_file.mtx> <n_columns_B>\n"; return 1; }
    std::string matrix_file = argv[1];
    int n = std::atoi(argv[2]);
    if (n <= 0) { std::cerr << "Error: columns > 0\n"; return 1; }

    auto A_csr = load_mtx_to_csr_canonical<int32_t,double,32>(matrix_file);
    size_t k = A_csr.ncols, m = A_csr.nrows;

    std::vector<double, AlignedAllocator<double,32>> B_val(k*n);
    for (size_t i=0;i<k;i++) for (int j=0;j<n;j++) B_val[i*n+j] = sin(i+j);

    // run once
    experiment_spmm_csr_simd(n, m, A_csr.row_ptr.data(), A_csr.col_idx.data(), A_csr.values.data(), B_val.data());

    std::cout << "Beginning SpMM CSR" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    auto C_csr = experiment_spmm_csr_simd(n,m,
                                          A_csr.row_ptr.data(),
                                          A_csr.col_idx.data(),
                                          A_csr.values.data(),
                                          B_val.data());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    std::cout << "spmm_csr time: "
              << std::fixed << std::setprecision(6)
              << (static_cast<double>(time_ns.count()) * 1e-9) << "s\n\n";

    auto C_mkl = experiment_spmm_mkl(n,m,k,
                                     A_csr.row_ptr.data(),
                                     A_csr.col_idx.data(),
                                     A_csr.values.data(),
                                     B_val.data());


    auto C_csb = experiment_spmm_csb(n, A_csr);

    std::cout << "CSR vs MKL match: " << (compare_dense_matrices(C_csr,C_mkl,m,n) ? "YES" : "NO") << "\n";
    std::cout << "CSR vs CSB match: " << (compare_dense_matrices(C_csr,C_csb,m,n) ? "YES" : "NO") << "\n";
    std::cout << "MKL vs CSB match: " << (compare_dense_matrices(C_mkl,C_csb,m,n) ? "YES" : "NO") << "\n";

    return 0;
}
