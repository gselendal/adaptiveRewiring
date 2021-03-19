module rewiring 

include("./utils.jl")

export getNodesReceivingAndSendingEdges, pickNodeWithInOutEdges, computeAdvectionKernel, computeConsensusKernel, computeKernel, computeKernel_reverse,runDynamics, runInDynamics, runOutDynamics, pruneInDegree, pruneOutDegree, rewireOutDegree, pruneInAndOutDegree, rewireInAndOutDegree
using LinearAlgebra, .utils, Random



function getNodesReceivingAndSendingEdges(A)
     """
    Get nodes both receiving (incoming) at least one edge and sending at least one edge
    and with un number of incoming and outgoing edges below the number of nodes
    in the network
    args:
        A
    returns:
        nodesReceivingAndSending
    """
    
    numNodes = size(A,1) # num of nodes 
    
    degIn = sum(A .!= 0, dims = 1) # num of incoming edges to the ith node
    
    degOut = transpose(sum(A .!= 0, dims = 2)) # num of outgoing edges to the ith node
    
    nodesReceiving = map(x -> x[2], findall(x -> (x .> 0) && (x .< (numNodes -1)), degIn)) # get nodes with in-degrees between 0 and numNodes
    
    nodesSending = map(x -> x[2], findall(x -> (x .> 0) && (x .< (numNodes -1)), degOut)) # get nodes with out-degrees between 0 and numNodes
    
    nodesReceivingAndSending = intersect(nodesReceiving, nodesSending)
    
    return nodesReceivingAndSending
end


function pickNodeWithInOutEdges(A, tau, pRandRewire)
 """
    Chose a node at random, keep a record of that node and other nodes
    args
        A: ?
            Adjacency matrix at initial state
        tau: ?
            heat dispersion parameter
        p_rnd_rewire: ?
            probability of random rewiring
    returns
        not_x: ?
        node_x: ?
    """
    
    numNodes = size(A,1)
    nodesReceivingAndSending = getNodesReceivingAndSendingEdges(A)
    
    if nodesReceivingAndSending == nothing
        return A
    end
    
    # else randomly pick a node
    ind = rand(1:length(nodesReceivingAndSending))
    nodeX = nodesReceivingAndSending[ind]
    
    # keep a record of the other nodes
    indAll = collect(1:numNodes)
    notX = filter(x -> !(x in nodeX), indAll)
    
    return (notX, nodeX)
end

function computeKernel(w, layer_dims)
"""
    Computes adjacency matrix which will be fed into rewiring algorithm, contains traffic between i-th and j-th nodes at each ij. 
            INPUT
    w: model, array containing all weights and biases
    layer_dims: array containing length of each layer
            OUTPUT
    adj_m: adjacency matrix with size N x N where N is number of nodes 
"""
    w = transpose.(w)

    N = sum(layer_dims)
    
    row = 1 
    col = layer_dims[1]+1
    
    A = Array{Float64, 2}(undef,N, N)
    A -= A
    
    for L=1:length(w)
        A[row:row+layer_dims[L]-1, col:col+layer_dims[L+1]-1] = w[L][:,:]
        row += layer_dims[L]; col += layer_dims[L+1]; 
    end
    
    #println(A)
    return transpose.(A)
end

function computeKernel_reverse(w, A, layer_dims)
"""
    Computes adjacency matrix which will be fed into rewiring algorithm, contains traffic between i-th and j-th nodes at each ij. 
            INPUT
    w: model, array containing all weights and biases
    layer_dims: array containing length of each layer
            OUTPUT
    adj_m: adjacency matrix with size N x N where N is number of nodes 
"""
    
    N = sum(layer_dims)
    
    row = 1 
    col = 1+layer_dims[1]
    for L=1:length(w)
        w[L][:,:] = transpose(A[row:row+layer_dims[L]-1, col:col+layer_dims[L+1]-1])
        row += layer_dims[L]; col += layer_dims[L+1]; 
    end
    
    return w
end


function computeConsensusKernel(A, tau)
    """
Calculate the consensus kernel
    Args:
        A:
            initial adjacency matrix
        tau:
            heat dispersion parameter
    """
    # estimate the in-degree Laplacian 
    n = size(A,1)
    
    # println("n: ", n)
    #println("sum: ", size(sum(A, dims = 1)))

    
    Din = sum(A, dims = 1) .* Matrix(I, n, n)
    
    #println("Din: ", size(Din))
    
    Lin = Din .- A

    # calculate the consensus kernel
    kernel = exp(-tau .* Lin)
    
    return kernel
end

function computeAdvectionKernel(A, tau)
    """
    Calculate the advection kernel. Use Lout instead of Lin (consensus case)
    Args:
        A:
            initial adjacency matrix
        tau:
            heat dispersion parameter
    """
    n = size(A,1)
    Dout = sum(A, dims = 2) .* Matrix(I, n, n)
    Lout = Dout .- A

    # calculate the consensus kernel
    kernel = exp(-tau .* Lout)
    
    return kernel
end



function pruneInAndOutDegree()
    println("..")
end 

function rewireInAndOutDegree()
    println("..")
end

function pruneOutDegree(AInit, pRandPrune, tau)
    A = copy(AInit)
    numNodes = size(A, 1)
    notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandPrune)
    
        if pRandPrune <= rand()
            # get node x's coldest head (excluding "x")
            xHeads = findall(x -> x != 0, A[:, nodeX])
            xCutHead = xHeads[argmin(A[xHeads, nodeX])]
        else # else we randomly rewire    
            xHeads = findall(x -> x != 0, A[:, nodeX])
            ind = rand(1:length(xHeads))
            xCutHead = xHeads[ind]
        end
        
        o = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = 0 

  
    return A, o, xCutHead, nodeX
end

function pruneInDegree(AInit, pRandPrune, tau)
        """
    Prunes iteratively a matrix A. At each iteration the pruning can be done selectively, lowest connection is pruned, (probability = 1 - pRandPrune) or 
    randomly (probability = pRandPrune). 

    Args:
        AInit:
            initial adjacency matrix
    Returns:
        A:
            
    """
    A = copy(AInit)
    numNodes = size(A, 1)
    
        
    notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandPrune)
                
    if pRandPrune <= rand()
        #println("1")
        # get node x's coldest tail (excluding "x")
        xTails = findall(x -> x != 0, A[nodeX, :])
        xCutTail = xTails[argmin(A[nodeX, xTails])]
    else # else we randomly prune    
        #println("2")
        xTails = findall(x -> x != 0, A[nodeX, :])
        ind = rand(1:length(xTails))
        xCutTail = xTails[ind]
    end
    
    o = A[nodeX, xCutTail]
    A[nodeX, xCutTail] = 0 
    
    
    return A, o, xCutTail, nodeX 
end

function rewireOutDegree(AInit, o, xCutHead, nodeX, pRandRewire)
        A = copy(AInit)
        numNodes = size(A, 1)

        # keep a record of the other nodes
        indAll = collect(1:numNodes)
        notX = filter(x -> !(x in nodeX) && !(x == xCutHead), indAll)

        # find the nodes with no incomings from "x"
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        if pRandRewire <= rand()
            xNonHeads = notX[findall(x -> x != 0, xNonHeadBool)]
            xWireNonHead = xNonHeads[argmax(A[xNonHeads, nodeX])]
        else # else we randomly rewire    
            xNonHeadBoolNonzero = findall(x -> x != 0, xNonHeadBool)
            ind = rand(1:length(xNonHeadBoolNonzero))
            xWireNonHead = notX[xNonHeadBoolNonzero[ind]]
        end
        
        if xCutHead == xWireNonHead
            println("PROBLEM")
            println("The A nodes rewired are $xWireNonHead and $nodeX with weight $A[xCutHead, nodeX]")
            println("The A nodes disconnected are $xCutHead and $nodeX")
        end
    
        A[xWireNonHead, nodeX] = o
  
    return A    
end

function rewireInDegree(AInit, o, xCutTail, nodeX, pRandRewire)
       
    A = copy(AInit)
    numNodes = size(A, 1)
    
        
    # keep a record of the other nodes
    indAll = collect(1:numNodes)
    notX = filter(x -> !(x in nodeX) && !(x == xCutTail), indAll)

    # find the nodes with no incomings from "x"
        
    xNonTailBool = 1 .* Bool.(A[nodeX, notX] .== 0) 
        
    if pRandRewire <= rand()
        # get node x's hottest non-tail (excluding "x")
        xNonTails = notX[findall(x -> x != 0, xNonTailBool)]
        xWireNonTail = xNonTails[argmax(A[nodeX, xNonTails])]            
    else # else we randomly rewire    
        xNonTailBoolNonzero = findall(x -> x != 0, xNonTailBool)
        ind = rand(1:length(xNonTailBoolNonzero))
        xWireNonTail = notX[xNonTailBoolNonzero[ind]]
    end
        
    if xCutTail == xWireNonTail
        println("PROBLEM")
        println("The A nodes rewired are $xWireNonTail and $nodeX with weight $A[nodeX, xCutTail]")
        println("The A nodes disconnected are $xCutTail and $nodeX")
    end
    
    A[xWireNonTail, nodeX] = o

    
    return A
end

function runInDynamics(AInit, pRandRewire, tau)
    
    A = copy(AInit)
    numNodes = size(A, 1)
    
        
    notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)

    # find the nodes with no incomings from "x"
        
    xNonTailBool = 1 .* Bool.(A[nodeX, notX] .== 0) 
        
    if pRandRewire <= rand()
        xTails = findall(x -> x != 0, A[nodeX, :])
        xCutTail = xTails[argmin(A[nodeX, xTails])]
   
        # get node x's hottest non-tail (excluding "x")
        xNonTails = notX[findall(x -> x != 0, xNonTailBool)]
        xWireNonTail = xNonTails[argmax(A[nodeX, xNonTails])]            
    else # else we randomly rewire 
        xTails = findall(x -> x != 0, A[nodeX, :])
        ind = rand(1:length(xTails))
        xCutTail = xTails[ind]
        
        xNonTailBoolNonzero = findall(x -> x != 0, xNonTailBool)
        ind = rand(1:length(xNonTailBoolNonzero))
        xWireNonTail = notX[xNonTailBoolNonzero[ind]]
    end
        
    if xCutTail == xWireNonTail
        println("PROBLEM")
        println("The A nodes rewired are $xWireNonTail and $nodeX with weight $A[nodeX, xCutTail]")
        println("The A nodes disconnected are $xCutTail and $nodeX")
    end
    
    A[xWireNonTail, nodeX] = A[nodeX, xCutTail]
    A[nodeX, xCutTail] = 0 
    
    return A
end

function runOutDynamics(AInit, pRandRewire, tau; rewire = true)
        A = copy(AInit)
        numNodes = size(A, 1)

        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)

        # find the nodes with no incomings from "x"
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        if pRandRewire <= rand()
            xHeads = findall(x -> x != 0, A[:, nodeX])
            xCutHead = xHeads[argmin(A[xHeads, nodeX])]
            
            if rewire == true
                xNonHeads = notX[findall(x -> x != 0, xNonHeadBool)]
                xWireNonHead = xNonHeads[argmax(A[xNonHeads, nodeX])]
            end
        else # else we randomly rewire    
            xHeads = findall(x -> x != 0, A[:, nodeX])
            ind = rand(1:length(xHeads))
            xCutHead = xHeads[ind]
            
            if rewire == true
                xNonHeadBoolNonzero = findall(x -> x != 0, xNonHeadBool)
                ind = rand(1:length(xNonHeadBoolNonzero))
                xWireNonHead = notX[xNonHeadBoolNonzero[ind]]
            end
        end
        if rewire == true
            println("Rewiring: ($xWireNonHead,$nodeX) with ($xCutHead, $nodeX)")
            A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        end
        println("Pruning: ($xCutHead, $nodeX)")
        A[xCutHead, nodeX] = 0 
    return A  
end

function runDynamics(A, p, tau, direction)
    
    if direction == "outdegree"
        A, o, xCutHead, nodeX = pruneOutDegree(A, p, tau)
        A = rewireOutDegree(A, o, xCutHead, nodeX, p)
        
    elseif direction == "indegree"
        A, o, xCutTail, nodeX = pruneInDegree(A, p, tau)
        A = rewireInDegree(A, o, xCutTail, nodeX, p)
    else 
        A, o1, xCutHead, node1 = pruneOutDegree(A, p, tau)
        A, o2, xCutTail, node2 = pruneInDegree(A, p, tau)

        A = rewireOutDegree(A, o1, xCutHead, node1, p)
        A = rewireInDegree(A, o2, xCutTail, node2, p)
    end
    
    return A 
end
end