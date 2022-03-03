module TestCore

import AtomixCUDA

using Atomix
using CUDA
using CUDA: @allowscalar
using Test

using ..Utils: cuda

# Not implemented:
#=
function test_get_set()
    A = CUDA.ones(Int, 3)
    cuda() do
        GC.@preserve A begin
            lens = Atomix.IndexLens((1,))
            x = Atomix.get(A, lens)
            Atomix.set!(A, lens, -x)
        end
    end
    @test collect(A) == [-1, 1, 1]
end
=#

function test_cas()
    idx = (
        data = 1,
        cas1_ok = 2,
        cas2_ok = 3,
        # ...
    )
    @assert minimum(idx) >= 1
    @assert maximum(idx) == length(idx)

    A = CUDA.zeros(Int, length(idx))
    cuda() do
        GC.@preserve A begin
            lens = Atomix.IndexLens((1,))
            (old, success) = Atomix.replace!(A, lens, 0, 42)
            A[idx.cas1_ok] = old == 0 && success
            (old, success) = Atomix.replace!(A, lens, 0, 43)
            A[idx.cas2_ok] = old == 42 && !success
        end
    end
    @test collect(A) == [42, 1, 1]
end

function test_inc()
    A = CUDA.CuVector(1:3)
    cuda() do
        GC.@preserve A begin
            lens = Atomix.IndexLens((1,))
            pre, post = Atomix.modify!(A, lens, +, 1)
            A[2] = pre
            A[3] = post
        end
    end
    @test collect(A) == [2, 1, 2]
end

end  # module
