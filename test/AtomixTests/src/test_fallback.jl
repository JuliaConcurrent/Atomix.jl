module TestFallback

using Atomix: @atomic, @atomicreplace, @atomicswap
using Test

mutable struct Atomic{T}
    @atomic x::T
end

function test()
    a = Atomic(123)
    @test (@atomic a.x) == 123
    @test (@atomic :monotonic a.x) == 123
    @atomic a.x = 456
    @test (@atomic a.x) == 456
    @atomic :monotonic a.x = 123
    @test (@atomic a.x) == 123
    @test (@atomic a.x += 111) == 234
    @test (@atomic :monotonic a.x += 111) == 345
    @test (@atomic a.x + 111) == (345 => 456)
    @test (@atomic :monotonic a.x + 111) == (456 => 567)
    @test (@atomicswap a.x = 123) == 567
    @test (@atomicswap :monotonic a.x = 234) == 123
    @test (@atomicreplace a.x 234 => 123) == (old = 234, success = true)
    @test (@atomicreplace a.x 234 => 123) == (old = 123, success = false)
    @test (@atomicreplace :monotonic a.x 123 => 234) == (old = 123, success = true)
    @test (@atomicreplace :monotonic a.x 123 => 234) == (old = 234, success = false)
    @test (@atomicreplace :monotonic :monotonic a.x 234 => 123) ==
          (old = 234, success = true)
    @test (@atomicreplace :monotonic :monotonic a.x 234 => 123) ==
          (old = 123, success = false)
end

end  # module
