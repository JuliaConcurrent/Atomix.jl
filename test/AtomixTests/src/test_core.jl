module TestCore

using Atomix
using Test

function test()
    A = ones(Int, 3)
    lens = Atomix.IndexLens((1,))
    @test Atomix.get(A, lens) === 1
    Atomix.set!(A, lens, 123)
    @test Atomix.get(A, lens) === 123
    @test Atomix.modify!(A, lens, +, 1) === (123, 124)
    @test Atomix.get(A, lens) === 124
    @test Atomix.replace!(A, lens, 124, 567) === (old = 124, success = true)
    @test Atomix.replace!(A, lens, 124, 567) === (old = 567, success = false)
end

end  # module
