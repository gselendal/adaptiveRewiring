module utils 
export checkIsNullException

function nonzero(a)
    b = []
    for i = 1:endof(a)
        if a[i] != 0 
            push!(b, i)
        end
    end
    
    return b 
end

end