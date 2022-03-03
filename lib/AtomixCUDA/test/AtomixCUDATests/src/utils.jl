module Utils

using CUDA

function cuda(f)
    function g()
        f()
        nothing
    end
    CUDA.@cuda g()
end

end  # module
