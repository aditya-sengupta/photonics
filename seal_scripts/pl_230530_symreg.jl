using SymbolicRegression
using NPZ

X = npzread("data/pl_230530/inputzs_230530_1514.npy")[:,1]
y = npzread("data/pl_230530/pl_intensities_230530_1514.npy")'

relu(x) = max(x, zero(x))

options = SymbolicRegression.Options(
    binary_operators=[+, *, -, /],
    unary_operators=[tanh],
    npopulations=20
)

hall_of_fame = EquationSearch(
    y, X, niterations=80, options=options,
    parallelism=:multithreading
)