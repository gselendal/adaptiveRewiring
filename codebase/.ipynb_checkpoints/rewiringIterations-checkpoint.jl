module rewiringIterations
include("./rewiring.jl")
include("./swnMetrics.jl")

using .rewiring, .swnMetrics, LinearAlgebra, Random 

export runDynamicsSteps, runDynamicsDiffValues, runDynamicsIterations

function runDynamicsSteps(numNodes, edges, pRandRewire, tau, numRewiringsVec, flagRewireMethod, flagAlg, weightDist = "binary", mu = nothing, sig = nothing)
     """
    Rewires iteratively an adjacency matrix and stores the rewired adjacency matrices at the dictionary A.
    The rewirings for which it stores the adjacency matrices are indicated by numRewiringsVec
    The method used by flagRewireMethod, the algorithm by flagAlg
    Args:
        AInit:
            initial adjacency matrix
        pRandRewire:
            probability of random rewiring
        tau:
            heat dispersion parameter
        numRewiringsVec:
            vector of the number of rewirings for which adjacency matrix is stored
        flagRewireMethod:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flagAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a dictionary with rewired adjacency matrices at different stages, i.e. A[4000] is adj. matrix after 4000 rewirings
    """
    
    AInit = swnMetrics.generateRandAdj(numNodes, edges, weightDist, mu, sig)
    A = Array{Any, 1}(nothing, len(numRewiringsVec))
    A[1] = AInit
    
    for (ind, numRewiringsVec) in enumerate(numRewiringsVec)
        if ind == 1 
            AInit = A[1]
            subtract = 0
        else 
            subtract = numRewiringsVec[ind - 1]
        end
            
        rewirings = numRewirings - subtract
        if flagRewireMethod == "in_out"
            A[numRewirings] = rewiring.runInAndOutDynamics(AInit, pRandRewire, rewirings, tau, flagAlg)
            
        elseif flagRewireMethod == "in"
            A[numRewirings] = rewiring.runInDynamics(AInit, pRandRewire, rewirings, tau, flagAlg)
            
        elseif flagRewireMethod == "out"
            A[numRewirings] = rewiring.runOutDynamics(AInit, pRandRewire, rewirings, tau, flagAlg)
        end
    # initialize for the next iteration
    AInit = A[numRewirings]
    end
    
    return A 
end

function runDynamicsDiffValues(numNodes, edges, pRandRewireVec, tauVec, numRewiringsVec, flagRewireMethod, flagAlg, weightDist = "binary", mu = nothing, sig = nothing)
      """
    Same as run_dynamics_steps but stores in dictionary for different tau and p_random values
    A[p_random,tau][num_rewirings] shows the adjacency matrix for p_random and tau values at num_rewirings
    The rewirings for which it stores the adjacency matrices are indicated by n_rewirings_vec
    The method used by flag_rewire_method, the algorithm by flag_alg
    Args:
        AInit:
            initial adjacency matrix
        pRandRewireVec:
            vector of the probabilities of random rewiring
        tauVec:
            vector of the tau values
        numRewiringsVec:
            vector of the number of rewirings for which adjacency matrix is stored
        flagRewireMethod:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flagAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a dictionary with rewired adjacency matrices
    """
    A = Dict()
    
    for p in pRandRewireVec
        for tau in tauVec
               A[(p, tau)] = runDynamicsSteps(numNodes, edges, p, tau, numRewiringsVec, flagRewireMethod, flagAlg, weightDist, mu, sig)
        end
    end
    
    return A
end

function runDynamicsIterations(numNodes, edges, pRandRewireVec, tauVec, numRewiringsVec, flagRewireMethod, flagAlg, iterations, weightDist = "binary", mu = nothing, sig = nothing)
        """
    Same as run_consensus_diff_values but for many iterations
    A_all[i][p_random,tau][num_rewirings] shows the i-th iteration adjacency matrix for p_random and tau values at num_rewirings
    The rewirings for which it stores the adjacency matrices are indicated by n_rewirings_vec
    The method used by flag_rewire_method
    Args:
        AInit:
            initial adjacency matrix
        pRandRewireVec:
            vector of the probabilities of random rewiring
        tauVec:
            vector of the tau values
        numRewiringsVec:
            vector of the number of rewirings for which adjacency matrix is stored
        flagRewireMethod:
            'in' or 'out' or 'in_out' depending on which consensus algorithm we use
        flagAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
        iterations:
            number of repetitions we run the different parameters
    Returns:
        A_all:
            returns a dictionary with rewired adjacency matrices
    """
    A_all = Dict()
    
    for i in collect(1:iterations)
        println("W are at $i iteration")
        A_all[i] = runDynamicsDiffValues(numNodes, edges, pRandRewireVec, tauVec, numRewiringsVec, flagRewireMethod, flagAlg, weightDist, mu, sig)
    end
    
    return A_all
end
end