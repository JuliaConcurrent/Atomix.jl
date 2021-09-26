module AtomicArraysTests

include("test_core.jl")
include("test_doctest.jl")

if isdefined(Base, :replaceproperty!)  # 1.7 or later
    include("test_properties.jl")
end

end  # module AtomicArraysTests
