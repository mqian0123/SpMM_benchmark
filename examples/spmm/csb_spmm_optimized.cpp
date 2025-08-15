#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <omp.h>
#include <immintrin.h> // For AVX/AVX2 intrinsics

// Structure to represent a non-zero element in the sparse matrix
struct NonZero {
    int row;
    int col;
    double value;
    
    NonZero(int r, int c, double v) : row(r), col(c), value(v) {}
    
    // For sorting by row-major order
    bool operator<(const NonZero& other) const {
        if (row != other.row)
            return row < other.row;
        return col < other.col;
    }
};

// CSB Block structure
struct CSBBlock {
    int row_start, row_end;
    int col_start, col_end;
    std::vector<NonZero> elements;
    
    // Compressed row indices and pointers
    std::vector<int> row_ptr;
    std::vector<int> row_idx;
    
    // Compressed column indices and pointers
    std::vector<int> col_ptr;
    std::vector<int> col_idx;
    
    // Store values in a separate array for better SIMD access
    std::vector<double> values;
    
    CSBBlock(int rs, int re, int cs, int ce) 
        : row_start(rs), row_end(re), col_start(cs), col_end(ce) {}
    
    void compress() {
        // Sort elements by row for row compression
        std::sort(elements.begin(), elements.end(), [](const NonZero& a, const NonZero& b) {
            if (a.row != b.row) return a.row < b.row;
            return a.col < b.col;
        });
        
        // Compress rows
        row_ptr.push_back(0);
        int current_row = -1;
        
        // Extract values into a separate array for better SIMD access
        values.reserve(elements.size());
        
        for (const auto& nz : elements) {
            if (nz.row != current_row) {
                current_row = nz.row;
                row_idx.push_back(current_row - row_start);
                row_ptr.push_back(row_idx.size() - 1);
            }
            col_idx.push_back(nz.col - col_start);
            values.push_back(nz.value);
        }
        row_ptr.push_back(col_idx.size());
        
        // Sort elements by column for column compression
        std::sort(elements.begin(), elements.end(), [](const NonZero& a, const NonZero& b) {
            if (a.col != b.col) return a.col < b.col;
            return a.row < b.row;
        });
        
        // Compress columns
        col_ptr.push_back(0);
        int current_col = -1;
        
        for (const auto& nz : elements) {
            if (nz.col != current_col) {
                current_col = nz.col;
                col_ptr.push_back(col_ptr.back() + 1);
            }
        }
    }
};

// Memory-aligned allocator for better SIMD performance
template <typename T, size_t Alignment = 32> // 32 bytes for AVX/AVX2
class AlignedAllocator {
public:
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    
    T* allocate(size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T))) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* p, size_t) {
        free(p);
    }
};

// CSB Matrix class
class CSBMatrix {
private:
    int rows, cols;
    int block_size;
    std::vector<std::vector<CSBBlock>> blocks;
    std::vector<double, AlignedAllocator<double>> values; // Aligned for SIMD

public:
    CSBMatrix(int r, int c, int bs) : rows(r), cols(c), block_size(bs) {
        int row_blocks = (rows + block_size - 1) / block_size;
        int col_blocks = (cols + block_size - 1) / block_size;
        
        blocks.resize(row_blocks);
        for (int i = 0; i < row_blocks; i++) {
            blocks[i].resize(col_blocks);
            for (int j = 0; j < col_blocks; j++) {
                int rs = i * block_size;
                int re = std::min(rs + block_size, rows);
                int cs = j * block_size;
                int ce = std::min(cs + block_size, cols);
                blocks[i][j] = CSBBlock(rs, re, cs, ce);
            }
        }
    }
    
    void addElement(int row, int col, double value) {
        int block_row = row / block_size;
        int block_col = col / block_size;
        blocks[block_row][block_col].elements.emplace_back(row, col, value);
        values.push_back(value);
    }
    
    void finalize() {
        // Use OpenMP to parallelize block compression
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < blocks.size(); i++) {
            for (int j = 0; j < blocks[i].size(); j++) {
                blocks[i][j].compress();
            }
        }
    }
    
    // Perform SpMM: C = A * B where A is this sparse matrix and B is dense
    std::vector<std::vector<double, AlignedAllocator<double>>> multiply(
        const std::vector<std::vector<double, AlignedAllocator<double>>>& B) const {
        
        if (cols != B.size()) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        int B_cols = B[0].size();
        std::vector<std::vector<double, AlignedAllocator<double>>> C(
            rows, std::vector<double, AlignedAllocator<double>>(B_cols, 0.0));
        
        // Parallelize the block-level multiplication
        #pragma omp parallel
        {
            // Create thread-local result matrix to avoid race conditions
            std::vector<std::vector<double, AlignedAllocator<double>>> C_local(
                rows, std::vector<double, AlignedAllocator<double>>(B_cols, 0.0));
            
            // Parallelize over blocks
            #pragma omp for collapse(2) schedule(dynamic)
            for (int bi = 0; bi < blocks.size(); bi++) {
                for (int bj = 0; bj < blocks[0].size(); bj++) {
                    const CSBBlock& block = blocks[bi][bj];
                    
                    // For each non-zero element in the block
                    for (size_t idx = 0; idx < block.values.size(); idx++) {
                        int i = block.row_idx[block.row_ptr[idx]] + block.row_start;
                        int j = block.col_idx[idx] + block.col_start;
                        double val = block.values[idx];
                        
                        // Use SIMD for the inner loop when possible
                        int k = 0;
                        
                        // Process 4 elements at a time using AVX
                        for (; k + 3 < B_cols; k += 4) {
                            __m256d b_vec = _mm256_loadu_pd(&B[j][k]);
                            __m256d c_vec = _mm256_loadu_pd(&C_local[i][k]);
                            __m256d val_vec = _mm256_set1_pd(val);
                            __m256d result = _mm256_fmadd_pd(val_vec, b_vec, c_vec);
                            _mm256_storeu_pd(&C_local[i][k], result);
                        }
                        
                        // Handle remaining elements
                        for (; k < B_cols; k++) {
                            C_local[i][k] += val * B[j][k];
                        }
                    }
                }
            }
            
            // Merge thread-local results
            #pragma omp critical
            {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < B_cols; j++) {
                        C[i][j] += C_local[i][j];
                    }
                }
            }
        }
        
        return C;
    }
    
    // Alternative implementation using row-wise parallelism
    std::vector<std::vector<double, AlignedAllocator<double>>> multiply_row_parallel(
        const std::vector<std::vector<double, AlignedAllocator<double>>>& B) const {
        
        if (cols != B.size()) {
            throw std::runtime_error("Matrix dimensions don't match for multiplication");
        }
        
        int B_cols = B[0].size();
        std::vector<std::vector<double, AlignedAllocator<double>>> C(
            rows, std::vector<double, AlignedAllocator<double>>(B_cols, 0.0));
        
        // Parallelize over rows of the result matrix
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < rows; i++) {
            int block_row = i / block_size;
            
            for (int bj = 0; bj < blocks[0].size(); bj++) {
                const CSBBlock& block = blocks[block_row][bj];
                
                // Find elements in this row
                for (size_t r = 0; r < block.row_ptr.size() - 1; r++) {
                    if (block.row_idx[r] + block.row_start == i) {
                        // Process all elements in this row
                        for (int ptr = block.row_ptr[r]; ptr < block.row_ptr[r + 1]; ptr++) {
                            int j = block.col_idx[ptr] + block.col_start;
                            double val = block.values[ptr];
                            
                            // Use SIMD for the inner loop
                            int k = 0;
                            
                            // Process 4 elements at a time using AVX
                            for (; k + 3 < B_cols; k += 4) {
                                __m256d b_vec = _mm256_loadu_pd(&B[j][k]);
                                __m256d c_vec = _mm256_loadu_pd(&C[i][k]);
                                __m256d val_vec = _mm256_set1_pd(val);
                                __m256d result = _mm256_fmadd_pd(val_vec, b_vec, c_vec);
                                _mm256_storeu_pd(&C[i][k], result);
                            }
                            
                            // Handle remaining elements
                            for (; k < B_cols; k++) {
                                C[i][k] += val * B[j][k];
                            }
                        }
                        break;
                    }
                }
            }
        }
        
        return C;
    }
    
    // Get matrix dimensions
    std::pair<int, int> getDimensions() const {
        return {rows, cols};
    }
    
    // Get number of non-zeros
    int getNonZeros() const {
        return values.size();
    }
};

// Function to read a Market Matrix file with parallel processing
CSBMatrix readMTXFile(const std::string& filename, int block_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::string line;
    bool header_found = false;
    int rows = 0, cols = 0, nnz = 0;
    
    // Skip comments and read header
    while (std::getline(file, line)) {
        if (line[0] == '%') continue;
        
        std::istringstream iss(line);
        iss >> rows >> cols >> nnz;
        header_found = true;
        break;
    }
    
    if (!header_found) {
        throw std::runtime_error("Invalid MTX file format: header not found");
    }
    
    // Create CSB matrix
    CSBMatrix matrix(rows, cols, block_size);
    
    // Read all non-zero elements into memory first
    std::vector<NonZero> elements;
    elements.reserve(nnz);
    
    int row, col;
    double value;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        if (iss >> row >> col >> value) {
            // MTX files are 1-indexed, convert to 0-indexed
            elements.emplace_back(row - 1, col - 1, value);
        }
    }
    
    // Add elements to the matrix in parallel
    #pragma omp parallel for schedule(dynamic, 1000)
    for (size_t i = 0; i < elements.size(); i++) {
        #pragma omp critical
        {
            matrix.addElement(elements[i].row, elements[i].col, elements[i].value);
        }
    }
    
    matrix.finalize();
    return matrix;
}

// Generate a random dense matrix with aligned memory
std::vector<std::vector<double, AlignedAllocator<double>>> generateRandomDenseMatrix(int rows, int cols) {
    std::vector<std::vector<double, AlignedAllocator<double>>> matrix(
        rows, std::vector<double, AlignedAllocator<double>>(cols));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Use thread-safe random number generation
            #pragma omp critical
            {
                matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
            }
        }
    }
    
    return matrix;
}

// Print a matrix (for small matrices)
void printMatrix(const std::vector<std::vector<double, AlignedAllocator<double>>>& matrix, 
                int max_rows = 10, int max_cols = 10) {
    int rows = std::min(max_rows, static_cast<int>(matrix.size()));
    int cols = std::min(max_cols, static_cast<int>(matrix[0].size()));
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i][j] << " ";
        }
        std::cout << (cols < matrix[0].size() ? "..." : "") << std::endl;
    }
    if (rows < matrix.size()) {
        std::cout << "..." << std::endl;
    }
}

// Function to run performance tests
void runPerformanceTests(const CSBMatrix& sparse_matrix, 
                        const std::vector<std::vector<double, AlignedAllocator<double>>>& dense_matrix) {
    auto dimensions = sparse_matrix.getDimensions();
    int dense_cols = dense_matrix[0].size();
    
    std::cout << "\n=== Performance Tests ===" << std::endl;
    
    // Test with different numbers of threads
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    for (int num_threads : thread_counts) {
        if (num_threads > omp_get_max_threads()) continue;
        
        omp_set_num_threads(num_threads);
        
        std::cout << "\nRunning with " << num_threads << " threads:" << std::endl;
        
        // Block-parallel implementation
        auto start = std::chrono::high_resolution_clock::now();
        auto result = sparse_matrix.multiply(dense_matrix);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::cout << "  Block-parallel SpMM: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                << " ms" << std::endl;
        
        // Row-parallel implementation
        start = std::chrono::high_resolution_clock::now();
        auto result2 = sparse_matrix.multiply_row_parallel(dense_matrix);
        end = std::chrono::high_resolution_clock::now();
        
        std::cout << "  Row-parallel SpMM: " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                << " ms" << std::endl;
        
        // Verify results match
        bool results_match = true;
        for (int i = 0; i < std::min(10, static_cast<int>(result.size())); i++) {
            for (int j = 0; j < std::min(10, static_cast<int>(result[0].size())); j++) {
                if (std::abs(result[i][j] - result2[i][j]) > 1e-10) {
                    results_match = false;
                    break;
                }
            }
            if (!results_match) break;
        }
        
        std::cout << "  Results match: " << (results_match ? "Yes" : "No") << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mtx_file> [block_size] [dense_matrix_cols] [num_threads]" << std::endl;
        return 1;
    }
    
    std::string mtx_file = argv[1];
    int block_size = (argc > 2) ? std::stoi(argv[2]) : 64;  // Default block size
    int dense_cols = (argc > 3) ? std::stoi(argv[3]) : 100; // Default number of columns for dense matrix
    int num_threads = (argc > 4) ? std::stoi(argv[4]) : omp_get_max_threads(); // Default to max available threads
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    try {
        std::cout << "Reading MTX file: " << mtx_file << std::endl;
        std::cout << "Using " << num_threads << " threads" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Read the sparse matrix
        CSBMatrix sparse_matrix = readMTXFile(mtx_file, block_size);
        auto dimensions = sparse_matrix.getDimensions();
        
        auto read_time = std::chrono::high_resolution_clock::now();
        std::cout << "Matrix dimensions: " << dimensions.first << " x " << dimensions.second << std::endl;
        std::cout << "Non-zeros: " << sparse_matrix.getNonZeros() << std::endl;
        std::cout << "Block size: " << block_size << std::endl;
        std::cout << "Time to read and convert to CSB: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(read_time - start_time).count() 
                  << " ms" << std::endl;
        
        // Generate a random dense matrix
        std::cout << "Generating random dense matrix of size " << dimensions.second << " x " << dense_cols << std::endl;
        auto dense_matrix = generateRandomDenseMatrix(dimensions.second, dense_cols);
        
        // Perform SpMM
        std::cout << "Performing SpMM..." << std::endl;
        auto mult_start = std::chrono::high_resolution_clock::now();
        auto result = sparse_matrix.multiply(dense_matrix);
        auto mult_end = std::chrono::high_resolution_clock::now();
        
        std::cout << "SpMM completed in " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(mult_end - mult_start).count() 
                  << " ms" << std::endl;
        
        // Print a small portion of the result
        std::cout << "Result matrix (showing up to 10x10):" << std::endl;
        printMatrix(result);
        
        // Run performance tests
        runPerformanceTests(sparse_matrix, dense_matrix);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}