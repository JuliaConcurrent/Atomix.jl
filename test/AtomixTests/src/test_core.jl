module TestCore

using AtomicArrays: AtomicRefArray
using Test
using UnsafeAtomics: UnsafeAtomics

function test()
    A = AtomicRefArray(ones(Int, 3))
    @test UnsafeAtomics.load(A[1]) === 1
    UnsafeAtomics.store!(A[1], 123)
    @test UnsafeAtomics.load(A[1]) === 123
    @test UnsafeAtomics.modify!(A[1], +, 1) === (123, 124)
    @test UnsafeAtomics.load(A[1]) === 124
    @test UnsafeAtomics.cas!(A[1], 124, 567) === (old = 124, success = true)
    @test UnsafeAtomics.cas!(A[1], 124, 567) === (old = 567, success = false)
end

end  # module
