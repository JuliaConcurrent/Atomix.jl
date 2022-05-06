module TestCore

using Atomix
using Test

function test()
    A = ones(Int, 3)
    ref = Atomix.IndexableRef(A, (1,))
    @test Atomix.get(ref) === 1
    Atomix.set!(ref, 123)
    @test Atomix.get(ref) === 123
    @test Atomix.modify!(ref, +, 1) === (123, 124)
    @test Atomix.get(ref) === 124
    @test Atomix.replace!(ref, 124, 567) === (old = 124, success = true)
    @test Atomix.replace!(ref, 124, 567) === (old = 567, success = false)
end

end  # module
