module TestProperties

using AtomicArrays: AtomicRefArray
using Test

function test()
    A = AtomicRefArray(ones(Int, 3))
    @atomic A[1]._ = 123
    @test (@atomic A[1]._) === 123
    @test (@atomic A[1]._ += 1) === 124
    @test (@atomic A[1]._) === 124
    @test (@atomicswap A[1]._ = 567) === 124
    @test (@atomic A[1]._) === 567
    @test (@atomicreplace A[1]._ 567 => 123) === (old = 567, success = true)
    @test (@atomicreplace A[1]._ 567 => 123) === (old = 123, success = false)
end

end  # module
