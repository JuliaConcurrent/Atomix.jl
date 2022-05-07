module TestCore

using Atomix
using Atomix.Internal: accessrecorder
using Test

function test_indexableref()
    A = ones(Int, 3)
    ref = Atomix.IndexableRef(A, (1,))
    @test eltype(ref) === Int
    @test Atomix.get(ref) === 1
    Atomix.set!(ref, 123)
    @test Atomix.get(ref) === 123
    @test Atomix.modify!(ref, +, 1) === (123, 124)
    @test Atomix.get(ref) === 124
    @test Atomix.replace!(ref, 124, 567) === (old = 124, success = true)
    @test Atomix.replace!(ref, 124, 567) === (old = 567, success = false)
end

function test_referenceablearray()
    @testset for a in Any[
        ones(Int, 3),
        view(ones(Int, 3), 1:2),
        view(ones(Int, 2, 3), 1:1, 1:2),
        ones(Int, 3)',
    ]
        ra = accessrecorder(a)
        @test size(ra) == size(a)
        @test IndexStyle(ra) == IndexStyle(a)
    end
end

end  # module
