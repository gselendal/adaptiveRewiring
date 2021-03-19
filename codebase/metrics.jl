module metrics
using LinearAlgebra
export net_density

function net_density(A)
    n = size(A,1)
    return sum(abs, map(i -> i != 0 ? 1 : 0, triu(A))) / (n * (n-1)) / 2
end
end