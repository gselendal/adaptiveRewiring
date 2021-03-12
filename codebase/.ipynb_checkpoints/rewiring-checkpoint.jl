module rewiring 

include("./utils.jl")

export getNodesReceivingAndSendingEdges, pickNodeWithInOutEdges, computeAdvectionKernel, computeConsensusKernel, runDynamicsAdvectionConsensus, runDynamicsAdvectionConsensusParallel, runDynamicsAdvectionConsensusSeq, runDynamicsAdvectionConsensusWithRandRewire, runInAndOutDynamics, runInDynamicsMLP,runOutDynamics, computeKernel, computeKernel_reverse, pruneInDegrees, pruneOutDegrees, rewireOutDegree
using LinearAlgebra, .utils, Random

function pruneOutDegrees(AInit, pRandPrune, tau)
    A = copy(AInit)
    numNodes = size(A, 1)
    
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandPrune)
        
        # calculate the consensus or advection kernel
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

function pruneInDegrees(AInit, pRandPrune, tau)
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
                
    # calculate the consensus or advection kernel
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
            
    A[nodeX, xCutTail] = 0 
    
    
    return A, xCutTail, nodeX 
end

function rewireOutDegree(AInit, o, xCutHead, nodeX, pRandRewire)
        A = copy(AInit)
        numNodes = size(A, 1)

        # keep a record of the other nodes
        indAll = collect(1:numNodes)
        notX = filter(x -> !(x in nodeX) && !(x == xCutHead), indAll)

        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        # calculate the consensus or advection kernel
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
    
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = 0 
  
    return A    
end

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


function runOutDynamics(AInit, pRandRewire, numRewirings, tau, flagAlg)
        """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random (probability = p_rnd_rewire) or according to a consensus or advection function (probability = 1-p_rnd_rewire). 
    
    It rewires only the OUTDEGREES. More specifically, during each rewiring iteration a random node k is selected and one of its outdegrees is cut, and a connection is added with the tail being k. It operates in the columns Works for both binary and weighted initial networks since this implementation just redistributes the weights
    
    Args:
        AInit:
            initial adjacency matrix
        pRandRewire:
            probability of random rewiring
        numRewirings:
            number of rewiring iterations
        tau:
            heat dispersion parameter
        flagAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a rewired matrix
    """
    
    A = copy(AInit)
    numNodes = size(A, 1)
    
    for i=1:numRewirings
        println("Iteration is $i")
        
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
        # find the nodes with no incomings from "x"
        
        xNonHeadBool = 1 .* Bool.(A[notX, nodeX] .== 0) 
        
        # calculate the consensus or advection kernel
        if pRandRewire <= rand()
            if flagAlg == "consensus"
                kernel = computeConsensusKernel(A,tau)
            elseif flagAlg == "advection"
                kernel = computeAdvectionKernel(A, tau)
            end
            
            # get node x's coldest head (excluding "x")
            xHeads = findall(x -> x != 0, A[:, nodeX])
            xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
            xNonHeads = notX[findall(x -> x != 0, xNonHeadBool)]
            xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
        else # else we randomly rewire    
            xHeads = findall(x -> x != 0, A[:, nodeX])
            ind = rand(1:length(xHeads))
            xCutHead = xHeads[ind]
            
            xNonHeadBoolNonzero = findall(x -> x != 0, xNonHeadBool)
            ind = rand(1:length(xNonHeadBoolNonzero))
            xWireNonHead = notX[xNonHeadBoolNonzero[ind]]
        end
        
        if xCutHead == xWireNonHead
            println("PROBLEM")
            println("The A nodes rewired are $xWireNonHead and $nodeX with weight $A[xCutHead, nodeX]")
            println("The A nodes disconnected are $xCutHead and $nodeX")
        end
          
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = 0 
    end
  
    return A     
end

function runInDynamicsMLP(AInit, pRandRewire, tau, flagAlg)
    """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random
    (probability = pRandRewire) or according to a consensus/advection function (probability = 1-pRandRewire).
    It rewires only the INDEGREES. More specifically, during each rewiring iteration a random node k is selected
    and one of its tails is cut, and a connection is added with the head being k. It operates in the rows
    Args:
        AInit:
            initial adjacency matrix
        pRandRewire:
            probability of random rewiring
        n_rewirings:
            number of iterations the wiring take place
        numRewirings:
            heat dispersion parameter
        flagAAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a rewired  matrix
    """

    A = copy(AInit)
    numNodes = size(A, 1)
    
        
    notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
    # find the nodes with no incomings from "x"
        
    xNonTailBool = 1 .* Bool.(A[nodeX, notX] .== 0) 
        
    # calculate the consensus or advection kernel
    if pRandRewire <= rand()
        if flagAlg == "consensus"
            kernel = computeConsensusKernel(A,tau)
        elseif flagAlg == "advection"
            kernel = computeAdvectionKernel(A, tau)
        end
            
        # get node x's coldest tail (excluding "x")
        xTails = findall(x -> x != 0, A[nodeX, :])
        xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
        # get node x's hottest non-tail (excluding "x")
        xNonTails = notX[findall(x -> x != 0, xNonTailBool)]
        xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
            
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
        
    A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
    A[nodeX, xCutTail] = 0 
    
    
    return A 
end

function runInAndOutDynamics(AInit, pRandRewire, numRewirings, tau, flagAlg)
    """
    Rewires iteratively a matrix A. At each iteration the rewiring can be random
    (probability= pRandRewire) or according to a consensus/advection function (probability = 1-pRandRewire).
    It rewires both the OUTDEGREES and INDEGREES
    Args:
        AInit:
            initial adjacency matrix
        pRandRewire:
            probability of random rewiring
        numRewirings:
            number of iterations the wiring take place
        tau:
            heat dispersion parameter
         flagAlg:
            'consensus' or 'advection' depending on which Laplacian you use for the kernel; Lin or Lout
    Returns:
        A:
            returns a rewired  matrix
    """
    
    A = copy(AInit)
    numNodes = size(A, 1)
    
    for i=1:numRewirings
        println("Iteration is $i")
        
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
            
                
        # Identify nodes that do not send an edge to "x
        xNonTailBool = 1.0 .* .!Bool.(A[nodeX, notX])
        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1.0 .* .!Bool.(A[notX, nodeX])
        
        
        # calculate the consensus or advection kernel
        if pRandRewire <= rand()
            if flagAlg == "consensus"
                kernel = computeConsensusKernel(A,tau)
            elseif flagAlg == "advection"
                kernel = computeAdvectionKernel(A, tau)
            end
            
            # ====== REWIRING FOR THE INCOMING EDGES OF THE NODE X ====
            # get node x's coldest tail (excluding "x")
            xTails = utils.nonzero(A[nodeX, :])[1]
            xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
            # get node x's hottest non-tail (excluding "x")
            xNonTails = notX[utils.nonzero(xNonTailBool)[1]]
            xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
            
            # ====== REWIRING FOR THE OUTGOING EDGES OF THE NODE X ====
            
            # get node x's coldest head (excluding "x")
            xHeads = utils.nonzero(A[:, nodeX])[1]
            xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
            xNonHeads = notX[utils.nonzero(xNonHeadBool)[1]]
            xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
        else # else we randomly rewire    
            
            ####################################
            # randomly pick one of x's tails (excluding "x")
            xTails = utils.nonzero(A[nodeX, :])[1]
            xCutTail = rand(1:xTails)
            
            # randomly pick one of x's non-tails (excluding "x")
            xWireNonTail = notX[rand(utils.nonzero(xNonTailBool)[1])]
            
            ##################################
            # randomly pick one of x's head (excluding "x")
            xHeads = utils.nonzero(A[:, nodeX])[1]
            xCutHead = rand(1:xHeads)
        
            xWireNonHead = notX[rand(utils.nonzero(xNonHeadBool)[1])]
            
        end
        
        if xCutTail == xWireNonTail || xCutHead == xWireNonHead
            println("PROBLEM")
        end
        
        A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
    

        A[nodeX, xCutTail] = 0 
        A[xCutHead, nodeX] = 0 
    end
    
    
    return A 
end

# either run advection or consensus at each iteration
function runDynamicsAdvectionConsensusSeq(AInit, numRewirings)      
    
    A = copy(AInit)
    numNodes = size(A, 1)
    pRandRewire = 0 
    
    for i=1:numRewirings
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
        # Identify nodes that do not send an edge to "x
        xNonTailBool = 1.0 .* .!Bool.(A[nodeX, notX])
        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1.0 .* .!Bool.(A[notX, nodeX])
        
        if i % 2 == 0
            println("Even iteration, we do consensus")
            
            # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====

            kernel = computeConsensusKernel(A,tau)
            
            # get node x's coldest tail (excluding "x")
            xTails = utils.nonzero(A[nodeX, :])[1]
            xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
            # get node x's hottest non-tail (excluding "x")
            xNonTails = notX[utils.nonzero(xNonTailBool)[1]]
            xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
            
            
            # cut and rewire in-going connections from node x
            A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
            A[nodeX, xCutTail] = 0 
        else 
            println("Odd iteration, we do advection")
            
            # ====== REWIRING FOR THE OUT-DEGREE OF NODE X =====
            
            kernel = computeAdvectionKernel(A, tau)
            
            
            # get node x's coldest head (excluding "x")
            xHeads = utils.nonzero(A[:, nodeX])[1]
            xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
            xNonHeads = notX[utils.nonzero(xNonHeadBool)[1]]
            xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
            
            # cut and rewire out-going connections from node x 
            A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
            A[xCutHead, nodeX] = 0 
        end
    end

    return A
end

function runDynamicsAdvectionConsensusParallel(AInit, numRewirings, tau)
    
    A = copy(AInit)
    numNodes = size(A, 1)
    pRandRewire = 0 
    
    for i=1:numRewirings
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
        # Identify nodes that do not send an edge to "x
        xNonTailBool = 1.0 .* .!Bool.(A[nodeX, notX])
        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1.0 .* .!Bool.(A[notX, nodeX])
        
        # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====

        kernel = computeConsensusKernel(A,tau)
        
                    
        # get node x's coldest tail (excluding "x")
        xTails = utils.nonzero(A[nodeX, :])[1]
        xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
        # get node x's hottest non-tail (excluding "x")
        xNonTails = notX[utils.nonzero(xNonTailBool)[1]]
        xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
        
        # ====== REWIRING FOR THE out-DEGREE OF NODE X =====
        
        kernel = computeAdvectionKernel(A, tau)
        
        # get node x's coldest head (excluding "x")
        xHeads = utils.nonzero(A[:, nodeX])[1]
        xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
        # Get x's hottest non-head (excluding "x")
        xNonHeads = notX[utils.nonzero(xNonHeadBool)[1]]
        xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
        
        ############################
        # cut and rewire in-going connections from node x
        A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
        A[nodeX, xCutTail] = 0 
        
        # cut and rewire out-going connections from node x 
        A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
        A[xCutHead, nodeX] = 0 
    end

    return A
end

# either run advection or consensus depending on pAdv. For example if pAdv = 0.7, every time there is a 70% probability that advection will take place, and a 30% that consensus will take place

function runDynamicsAdvectionConsensus(AInit, numRewirings, tau, pAdv)
    
    A = copy(AInit)
    numNodes = size(A, 1)
    pRandRewire = 0 
    
    for i=1:numRewirings
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
        # Identify nodes that do not send an edge to "x
        xNonTailBool = 1.0 .* .!Bool.(A[nodeX, notX])
        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1.0 .* .!Bool.(A[notX, nodeX])
        
        if pRandRewire <= rand()
            println("We do consensus")
            
            # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====

            kernel = computeConsensusKernel(A,tau)
            
            # get node x's coldest tail (excluding "x")
            xTails = utils.nonzero(A[nodeX, :])[1]
            xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
            # get node x's hottest non-tail (excluding "x")
            xNonTails = notX[utils.nonzero(xNonTailBool)[1]]
            xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
            
            
            # cut and rewire in-going connections from node x
            A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
            A[nodeX, xCutTail] = 0 
        else 
            println("We do advection")
            
            # ====== REWIRING FOR THE OUT-DEGREE OF NODE X =====
            
            kernel = computeAdvectionKernel(A, tau)
            
            
            # get node x's coldest head (excluding "x")
            xHeads = utils.nonzero(A[:, nodeX])[1]
            xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
            # Get x's hottest non-head (excluding "x")
            xNonHeads = notX[utils.nonzero(xNonHeadBool)[1]]
            xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
            
            # cut and rewire out-going connections from node x 
            A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
            A[xCutHead, nodeX] = 0 
        end
    end

    return A
end

function runDynamicsAdvectionConsensusWithRandRewire(AInit, numRewirings, tau, pAdv, pRandRewirings)
    
    A = copy(AInit)
    numNodes = size(A, 1)
    pRandRewire = 0 
    
    for i=1:numRewirings
        notX, nodeX = pickNodeWithInOutEdges(A, tau, pRandRewire)
        
        # Identify nodes that do not send an edge to "x
        xNonTailBool = 1.0 .* .!Bool.(A[nodeX, notX])
        
        # find the nodes with no incomings from "x"
        xNonHeadBool = 1.0 .* .!Bool.(A[notX, nodeX])
        
        if pRandRewire <= rand()
            if pAdv <= rand()
                println("We do consensus")
            
                # ====== REWIRING FOR THE in-DEGREE OF THE NODE X ====

                kernel = computeConsensusKernel(A,tau)
            
                # get node x's coldest tail (excluding "x")
                xTails = utils.nonzero(A[nodeX, :])[1]
                xCutTail = xTails[argmin(kernel[nodeX, xTails])]
            
                # get node x's hottest non-tail (excluding "x")
                xNonTails = notX[utils.nonzero(xNonTailBool)[1]]
                xWireNonTail = xNonTails[argmax(kernel[nodeX, xNonTails])]
            
            
                # cut and rewire in-going connections from node x
                A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
                A[nodeX, xCutTail] = 0 
            else
                println("We do advection")
                
                # ====== REWIRING FOR THE OUT-DEGREE OF NODE X =====
            
                kernel = computeAdvectionKernel(A, tau)
            
            
                # get node x's coldest head (excluding "x")
                xHeads = utils.nonzero(A[:, nodeX])[1]
                xCutHead = xHeads[argmin(kernel[xHeads, nodeX])]
            
                # Get x's hottest non-head (excluding "x")
                xNonHeads = notX[utils.nonzero(xNonHeadBool)[1]]
                xWireNonHead = xNonHeads[argmax(kernel[xNonHeads, nodeX])]
            
                # cut and rewire out-going connections from node x 
                A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
                A[xCutHead, nodeX] = 0 
            end
                
                
        else  # or randomly rewire
            ####################################
            # randomly pick one of x's tails (excluding "x")
            xTails = utils.nonzero(A[nodeX, :])[1]
            xCutTail = rand(1:xTails)
            
            # randomly pick one of x's non-tails (excluding "x")
            xWireNonTail = notX[rand(utils.nonzero(xNonTailBool)[1])]
            
            ##################################
            # randomly pick one of x's head (excluding "x")
            xHeads = utils.nonzero(A[:, nodeX])[1]
            xCutHead = rand(1:xHeads)
        
            xWireNonHead = notX[rand(utils.nonzero(xNonHeadBool)[1])]
            
            if xCutTail == xWireNonTail || xCutHead == xWireNonHead
                println("Problem")
            end
                
            A[nodeX, xWireNonTail] = A[nodeX, xCutTail]
            A[xWireNonHead, nodeX] = A[xCutHead, nodeX]
    

            A[nodeX, xCutTail] = 0 
            A[xCutHead, nodeX] = 0 
        end
    end
    return A
end

end