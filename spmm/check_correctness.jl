using MatrixMarket
using LinearAlgebra
using JSON3
using NPZ

if length(ARGS) != 3
    println("Usage: julia check_correctness.jl <input_file> <output_dir> <n>")
    exit(1)
end

input_dir = ARGS[1]
output_dir = ARGS[2]
n = parse(Int, ARGS[3])

# TODO: In the future, not reading from just .mtx files, need to generalize for all possible forms
A = MatrixMarket.mmread(input_dir)
m, k = size(A)

B = [sin(i + j) for i = 0:k-1, j = 0:n-1]
C_julia = A * B  # m × n matrix

# load output from output_dir 
meta = JSON3.read(read(joinpath(output_dir, "C.bspnpy", "binsparse.json"), String))

m_cpp, n_cpp = Tuple(meta["shape"])
C_vec = NPZ.npzread(joinpath(output_dir, "C.bspnpy", "values.npy"))
C_cpp = reshape(C_vec, (n_cpp, m_cpp))'  # transpose to get row-major shape, default column-major


tolerance = 1e-10
if isapprox(C_cpp, C_julia; atol=tolerance, rtol=tolerance)
    println("SpMM result is correct within tolerance (", tolerance, ")")
else
    println("SpMM result is incorrect")
    abs_diff = abs.(C_cpp .- C_julia)
    max_abs_err = maximum(abs_diff)
    rel_err = norm(C_cpp - C_julia) / norm(C_julia)
    println("Max abs error: ", max_abs_err)
    println("Relative error: ", rel_err)

end
