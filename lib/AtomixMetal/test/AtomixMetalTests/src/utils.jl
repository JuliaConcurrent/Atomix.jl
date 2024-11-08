module Utils

using Metal

function metal(f)
    function g()
        f()
        nothing
    end
    Metal.@metal g()
end

end  # module
