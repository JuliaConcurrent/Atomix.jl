module TestSugar

using AtomicArrays: @atomic
using Test

function test_get()
    A = [42]
    @test (@atomic A[1]) === 42
    @test (@atomic A[end]) === 42
end

function test_set()
    A = [42, 43]
    @atomic A[1] = 123
    @atomic A[end] = 124
    @test A[1] === 123
    @test A[end] === 124
end

function test_inc()
    A = [1, 1]
    @test (@atomic A[1] += 123) === 124
    @test A[1] === 124
    @test (@atomic A[end] += 123) === 124
    @test A[end] === 124
end

end  # module
