module TestSugar

import AtomicArraysCUDA

using AtomicArrays: @atomic
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
            x = @atomic A[begin]
            @atomic A[end-2] = -x
        end
    end
    @test collect(A) == [-1, 1, 1]
end
=#

function test_inc()
    A = CUDA.ones(Int, 3)
    cuda() do
        GC.@preserve A begin
            @atomic A[begin] += 1
       end
    end
    @test collect(A) == [2, 1, 1]
end

end  # module
