module TestSugar

using AtomicArrays: @atomic, @atomicreplace, @atomicswap
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

function test_swap()
    A = [1, 1]
    @test (@atomicswap A[1] = 123) === 1
    @test A[1] === 123
    @test (@atomicswap A[end] = 456) === 1
    @test A[end] === 456
end

function test_cas()
    A = [1, 1]
    @test (@atomicreplace A[1] 1 => 123) == (old = 1, success = true)
    @test A[1] === 123
    @test (@atomicreplace A[1] 1 => 456) == (old = 123, success = false)
    @test A[1] === 123
    @test (@atomicreplace A[end] 1 => 789) == (old = 1, success = true)
    @test A[end] === 789
end

end  # module
