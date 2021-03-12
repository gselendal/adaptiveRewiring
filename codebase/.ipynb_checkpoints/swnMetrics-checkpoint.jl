module swnMetrics
export generateBinaryRandSymAdj, generateWeightRandSymAdj

using LinearAlgebra, Random, Distributions

function generateRandAdj(vertices, edges; weightDist = "binary", mu = nothing, sig = nothing)
    """
Generate a directed random matrix
    Args:
        vertices:
            number of vertices
        edges:
            number of edges (nonzero entries on the random matrix)

        weightDistribution:
            'binary', 'normal' or 'lognormal'
        mu:
            the mu parameter, is not valid for binary
        sig:
            the sig parameter, is not valid for binary
    Returns:
        A:
            the vertices X vertices random digraph
    """
    epsilon = 0.05 
    
    maxConnections = Int(vertices * (vertices - 1))
    
    if edges > maxConnections || edges < 0 
        println("(generateRandAdj) EEddge number out of range.")
        return -1
    end
    
    # sample weights from a lognormal distribution
    if weightDist == "lognormal"
        # lognormal distribution
        if mu == nothing && sig == nothing
            mu, sig = 0.,1.
        end
        logDist = LogNormal(mu, sig)
        randWeights = rand(logDist, edges) 
        
    elseif weightDist == "normal"
        # normal distribution
        if mu == nothing && sig == nothing
            mu, sig = 0.,1.
        end
        dist = Normal(mu, sig)
        randWeights = rand(dist, edges)
        ind = findall(randWeights .< 0)
        randWeights[ind] .= epsilon
        
    elseif weightDist == "binary"    
        randWeights = ones(edges)
    end
    
    # Normalize weights such that their sum equals the number of edges

    if weightDist == "lognormal" || weightDist == "normal"
        normFactor = length(randWeights) / sum(randWeights)
        normRandWeights = randWeights .* normFactor
    else 
        normRandWeights = randWeights
    end
    
    # Get the indices of 1s of a matrix the same size as A with 1s everywhere except in the diagonal
    Aones = ones(vertices,vertices) .- Matrix(I, vertices, vertices)
    ind = findall(Aones .> 0)
    # Get a random sample of those indices (#edges)
    indRand = shuffle(ind)[1:edges]
    
    # construct empty A with size vertices x vertices 
    aTemp = zeros(vertices, vertices)
    aTemp[indRand] .= normRandWeights
    A = aTemp
    
    return A
end

function generateBinaryRandSymAdj(vertices, edges)
""" 
    INPUT
vertices: # of nodes (vertices)
edges: # of edges    
    OUTPUT
A: random symmetric vertices X vertices adj matrix with 0s at Aij, Aji if the i-j are not connected, 1 if they are

"""
 
    maxConnections = Int(vertices * (vertices - 1) / 2)
    
    if edges > maxConnections || edges < 0 
        println("The number of edges are not within the permitted range.")
        return -1
    end
    
    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = triu(ones(vertices, vertices)) - 1 * Matrix(I, vertices, vertices)
    ind = findall(upperTriangOnes .> 0)
    
    # Get a random sample of those indices (#edges)
    indRand = shuffle(ind)[1:edges]
    
    # construct empty A with size vertices x vertices 
    aTemp = zeros(vertices, vertices)
    aTemp[indRand] .= 1 
    
    A = aTemp + transpose(aTemp)
    return A
end

function generateWeightRandSymAdj(vertices, edges, weightDist)
""" 
    INPUT 
vertices: # of nodes (vertices)
edges: # of edges
weightDist: it can either be normal or lognormal
    OUTPUT 
A: random symmetric vertices X vertices adjacency matrix with 0s at Aij, Aji if the i-j are not connected, 0<w<1 if they are. The weights follow a distribution specified.
"""
    
    maxConnections = Int(vertices * (vertices - 1) / 2) # max connections of a network can have
    epsilon = 0.05
    
    if edges > maxConnections || edges < 0
        println("The # of edges are not within the permitted range.")
    end
    
    if weightDist == "lognormal"
        mu, sig = 0.,1.
        logDist = LogNormal(mu, sig)
        randWeights = rand(logDist, edges) 
    elseif weightDist == "normal"
        mu, sig = 1., 0.25
        dist = Normal(mu, sig)
        randWeights = rand(dist, edges)
        ind = findall(randWeights .< 0)
        randWeights[ind] .= epsilon
    end
    
    
    # Normalize so that the sum of the weights equals the # of edges
    normFactor = length(randWeights) ./ sum(randWeights)
    normRandWeights = randWeights .* normFactor
    
    # Get the indices of 1s of a matrix with 1s only on its upper triangular part
    upperTriangOnes = triu(ones(vertices, vertices)) - 1 * Matrix(I, vertices, vertices)
    ind = findall(upperTriangOnes .> 0)
    
    # Get a random sample of those indices (#edges)
    indRand = shuffle(ind)[1:edges]
    
    # construct empty A with size vertices x vertices 
    aTemp = zeros(vertices, vertices)
    aTemp[indRand] .= normRandWeights
    
    A = aTemp + transpose(aTemp)
end

end