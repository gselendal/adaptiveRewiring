module metrics
using LinearAlgebra
export net_density

function net_density(A)
    n = size(A,1)
    return sum(abs, filter(i -> i != 0, triu(A))) / (n * (n-1)) / 2
end
end