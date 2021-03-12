module centrality 
include("./swnHeatKernels.jl")

using .swnHeatKernels 

function getArandA(vertices, edges, tau, pRandRewire, rewirings, weightDist)
    if weightDist == "binary"
        println("...")
    else 
        println(".....")
    end
    
    A = swnHeatKernels.rewireHeatKernel(Arand, pRandRewire, rewirings, tau)
    
end

function getArandAmany(vertices, edges, taus, pRandRewires, rewirings, weightDist)
    dictMetrics = Dict()
    
    for t in taus
        for p in pRandRewires
            dictMetrics[(p, t, rewirings)] = getArandA(vertices, edges, t, p, rewirings, weightDist)
            println("Finished pRand = $p, tau = $t, rewirings = $rewirings")
        end
    end
    return dictMetrics
end

function getArandAManyIterations(vertices, edges, taus, pRandRewires, rewirings, weightDist, iterations)
    dictMetricsIterations = Dict{Integer,Integer}()
    
    for i = 1:iterations
        println(typeof(i))
        println("Iteration $i started")
        dictMetricsIterations[i] = getArandAmany(vertices, edges, taus, pRandRewires, rewirings, weightDist)
    end
    
    return dictMetricsIterations   
end
end