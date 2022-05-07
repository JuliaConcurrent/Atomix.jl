module AtomixTests

include("test_core.jl")
include("test_sugar.jl")
include("test_doctest.jl")

if VERSION >= v"1.7"
    include("test_fallback.jl")
end

end  # module AtomixTests
