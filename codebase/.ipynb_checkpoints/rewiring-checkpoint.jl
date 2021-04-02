module rewiring
include("./utils.jl")

export getNodesReceivingAndSendingEdges, pickNodeWithInOutEdges, computeAdvectionKernel, computeConsensusKernel, computeKernel, computeKernel_reverse, runInDynamics, runOutDynamics, rewireOutdegree
# pruneInDegree, pruneOutDegree, rewireOutDegree, pruneInAndOutDegree, rewireInAndOutDegree
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
    
    nodesReceiving = map(x -> x[2], findall(x -> (x .> 0), degIn)) # get nodes with in-degrees between 0 and numNodes
    
    nodesSending = map(x -> x[2], findall(x -> (x .> 0), degOut)) # get nodes with out-degrees between 0 and numNodes
    
    nodesReceivingAndSending = intersect(nodesReceiving, nodesSending)
    
    push!(nodesReceivingAndSending, size(A,1)) # tiny modification to get the output node 
    
    return nodesReceivingAndSending
end


function pickNodeWithInOutEdges(A, pRandRewire)
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

function pickNodes(A, layer_dims)
    """
    Takes adjacency matrix, selects a random node and return other nodes in the SAME layer  
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
    s, e = findOtherNodesInLayer(nodeX, layer_dims)
    indAll = collect(s:e)
    notX = filter(x -> !(x in nodeX), indAll)
    
    return (notX, nodeX)
end

function findOtherNodesInLayer(nodeX, layer_dims)
    """
    Takes a random node X, returns the start and end index of its layer
    """
    inds = [1]
    l = 0
    
    for k=2:length(layer_dims)
        push!(inds, inds[k-1]+layer_dims[k-1])
        if (nodeX >= inds[k-1] && nodeX < inds[k])
            return (inds[k-2],inds[k-1]-1)
        end
    end
    return (inds[end-1],inds[end]-1)
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

function rewireOutdegree(AInit, D, pRandRewire, layer_dims)
        A = copy(AInit)
        numNodes = size(A, 1)

        notX, nodeX = pickNodes(A, layer_dims)

        # find the nodes with no incomings from "x"
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        if sum(xNonHeadBool)==0
            return A
        end
    
        if pRandRewire <= rand()
            xHeads = findall(x -> x != 0, A[:, nodeX])
            # find the node that abs sum of incoming and outgoing dw's are max
            xCutHead = xHeads[argmin(A[xHeads, nodeX])]
            #xHeads[argmax(transpose(sum(D[xHeads,:], dims = 2))+sum(D[:,xHeads], dims=1))[2]]
            
            xNonHeads = notX[findall(x -> x != 0, xNonHeadBool)]
            # find the node that abs sum of incoming and outgoing dw's are min
            xWireNonHead = xNonHeads[argmin(transpose(sum(abs, D[xNonHeads,:], dims = 2))+sum(abs, D[:,xNonHeads], dims=1))[2]]
        else # else we randomly rewire    
            xHeads = findall(x -> x != 0, A[:, nodeX])
            ind = rand(1:length(xHeads))
            xCutHead = xHeads[ind]
            
            xNonHeadBoolNonzero = findall(x -> x != 0, xNonHeadBool)
            ind = rand(1:length(xNonHeadBoolNonzero))
            xWireNonHead = notX[xNonHeadBoolNonzero[ind]]

        end
        println("Rewiring: non-head ($xWireNonHead,$nodeX) with head ($xCutHead, $nodeX)")
    
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = 0
        return A  
end


"""function runInDynamics(AInit, pRandRewire, tau)
    
    A = copy(AInit)
    numNodes = size(A, 1)
    
        
    notX, nodeX = pickNodeWithInOutEdges(A, pRandRewire)

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
        
    
    temp = A[xWireNonTail, nodeX]
    A[xWireNonTail, nodeX] = A[nodeX, xCutTail]
    A[nodeX, xCutTail] = temp
    
    return A
end

function runOutDynamics(AInit, pRandRewire, tau; rewire = true)
        A = copy(AInit)
        numNodes = size(A, 1)

        notX, nodeX = pickNodeWithInOutEdges(A, pRandRewire)

        # find the nodes with no incomings from "x"
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        if pRandRewire <= rand()
            xHeads = findall(x -> x != 0, A[:, nodeX])
            xCutHead = xHeads[argmin(A[xHeads, nodeX])]
            
            xNonHeads = notX[findall(x -> x != 0, xNonHeadBool)]
            xWireNonHead = xNonHeads[argmax(A[xNonHeads, nodeX])]
        else # else we randomly rewire    
            xHeads = findall(x -> x != 0, A[:, nodeX])
            ind = rand(1:length(xHeads))
            xCutHead = xHeads[ind]
            
            xNonHeadBoolNonzero = findall(x -> x != 0, xNonHeadBool)
            ind = rand(1:length(xNonHeadBoolNonzero))
            xWireNonHead = notX[xNonHeadBoolNonzero[ind]]
        end
        println("Rewiring: non-head ($xWireNonHead,$nodeX) with head ($xCutHead, $nodeX)")
        temp = A[xWireNonHead, nodeX]
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = temp 
    return A  
end
"""

end

