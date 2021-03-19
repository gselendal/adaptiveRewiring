module mlp 
include("./rewiring.jl")
include("./metrics.jl")
using Knet, Plots, Statistics, LinearAlgebra, Random,.rewiring, .metrics
export pred, loss, gen_data, split_data, genbatch, backprop

# prediction and loss functions
σ(x) = relu(x)
layer(w,b,x) = σ.(w*x .+ b)
pred(w,b,x)=sigm.(w[4]*layer(w[3],b[3],layer(w[2],b[2],layer(w[1],b[1],x))).+b[4])
loss(w,b,x,y; λ = 0.0) = mean(abs2,pred(w,b,x)-y) + (λ/2)mean(sum.(abs,w))

# returns binary digits of an integer
binar(nbits,x) = digits(x, base=2, pad=nbits) |> reverse

function gen_data(n, ntest)
    testidx = randperm(2^n-1)[1:ntest]
    trainidx = setdiff(1:2^n-1,testidx)
    return trainidx, testidx
end

function split_data(trainidx, testidx, n,ntest)
    # split integers in (1,..,2ⁿ-1) to training and test sets
     # samples not in the test set
    ntrain = length(trainidx)

    xtst = zeros(Int64,n,ntest)
    for i=1:ntest
        xtst[:,i] = binar(n,testidx[i])
    end
    ytst = sum(xtst,dims=1).%2

    xtrn = zeros(Int64,n,ntrain)
    for i=1:ntrain
        xtrn[:,i] = binar(n,trainidx[i])
    end
    ytrn = sum(xtrn,dims=1).%2

    return xtrn, ytrn, xtst, ytst
end


# Generate a batch: integer array of size (n x nbatch)
function genbatch(trainidx, nbit,nbat, ntest)
    batchidx = shuffle(trainidx)[1:nbat]
    out = zeros(Int64,nbit,nbat)
    for i=1:nbat
        out[:,i] = binar(nbit,batchidx[i])
    end
    return out
end

function backprop_2(w, b, niter, trainidx, testidx, n, ntest, nbatch, layer_dims; rewire = false, λ= 0.0, LR = 0.1, p = 0.0)
    losstrn = []
    losstst = []    
    xtrn, ytrn, xtst, ytst = mlp.split_data(trainidx, testidx, n, ntest)
    rho = []
    tau = 4.5 
    
    t = 1 
    for k in progress(1:niter)
        xin = genbatch(trainidx, n,nbatch, ntest)
        yin = sum(xin,dims=1).%2
        dl = @diff loss(w,b,xin,yin; λ= λ)
        for i=1:length(w)
            w[i] .-= LR *grad(dl,w[i])
            b[i] .-= LR *grad(dl,b[i])
        end
        
        A = computeKernel(w,layer_dims)
        push!(rho, metrics.net_density(A))
        
        if k == 2^t 
            t += 1 
            
            A, o, xCutHead, nodeX = pruneOutDegree(A, p, tau)

            if rewire == true
                A = rewireOutDegree(A, o, xCutHead, nodeX, p)
            end
            
            w = computeKernel_reverse(w, A, layer_dims)
        end
        
        if (k%1000==1)
            xin = genbatch(trainidx, n,ntest, ntest) # training set samples with size equal to test set
            yin = sum(xin,dims=1).%2 
            push!(losstrn,loss(w,b,xin,yin; λ= λ)) # record loss over 1000 samples
            push!(losstst,loss(w,b,xtst,ytst; λ= λ))
        end
        
    end
    
    return w, b, losstrn, losstst, rho
end


function backprop(w, b, niter, trainidx, testidx, n, ntest, nbatch, layer_dims; rewire = false, λ= 0.0, LR = 0.1, p = 0.0)
    losstrn = []
    losstst = []    
    xtrn, ytrn, xtst, ytst = mlp.split_data(trainidx, testidx, n, ntest)
    rho = []
    tau = 4.5 
    
    t = 1 
    for k in progress(1:niter)
        xin = genbatch(trainidx, n,nbatch, ntest)
        yin = sum(xin,dims=1).%2
        dl = @diff loss(w,b,xin,yin; λ= λ)
        for i=1:length(w)
            w[i] .-= LR *grad(dl,w[i])
            b[i] .-= LR *grad(dl,b[i])
        end
         
        
        if k == 2^t 
            t += 1 
            
            A = computeKernel(w,layer_dims)

            A = runOutDynamics(A, p, tau, rewire = rewire)
           """ A, o, xCutHead, nodeX = pruneOutDegree(A, p, tau)

            if rewire == true
                A = rewireOutDegree(A, o, xCutHead, nodeX, p)
            end
            """
            w = computeKernel_reverse(w, A, layer_dims)
        end
        
        if (k%1000==1)
            xin = genbatch(trainidx, n,ntest, ntest) # training set samples with size equal to test set
            yin = sum(xin,dims=1).%2 
            push!(losstrn,loss(w,b,xin,yin; λ= λ)) # record loss over 1000 samples
            push!(losstst,loss(w,b,xtst,ytst; λ= λ))
        end
        
    end
    
    return w, b, losstrn, losstst, rho
end
end