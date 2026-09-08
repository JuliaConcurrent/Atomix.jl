using Metal
using Metal: @allowscalar


@testset "AtomixMetalExt:extension_found" begin
    @test !isnothing(Base.get_extension(Atomix, :AtomixMetalExt))
end


function metal(f)
    function g()
        f()
        nothing
    end
    Metal.@metal g()
end




@testset "AtomixMetalExt:test_cas" begin
    idx = (
        data = 1,
        cas1_ok = 2,
        cas2_ok = 3,
        # ...
    )
    @assert minimum(idx) >= 1
    @assert maximum(idx) == length(idx)

    A = Metal.zeros(Int32, length(idx))
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            (old, success) = Atomix.replace!(ref, 0, 42)
            A[idx.cas1_ok] = old == 0 && success
            (old, success) = Atomix.replace!(ref, 0, 43)
            A[idx.cas2_ok] = old == 42 && !success
        end
    end
    @test collect(A) == [42, 1, 1]
end


@testset "AtomixMetalExt:test_inc" begin
    A = Metal.MtlVector(Int32(1):Int32(3))
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            pre, post = Atomix.modify!(ref, +, 1)
            A[2] = pre
            A[3] = post
        end
    end
    @test collect(A) == [2, 1, 2]
end

@testset "AtomixMetalExt:test_inc_threadgroup" begin
    A = Metal.MtlVector(Float32(1):Float32(3))
    metal() do
        GC.@preserve A begin
            B = Metal.MtlThreadGroupArray(Float32, 3)
            B[1] = A[1]
            refB = Atomix.IndexableRef(B, (1,))
            pre, post = Atomix.modify!(refB, +, Float32(1))
            A[1] = B[1]
            A[2] = pre
            A[3] = post
        end
    end
    @test collect(A) == Float32[2, 1, 2]
end


@testset "AtomixMetalExt:test_inc_sugar" begin
    A = Metal.ones(Int32, 3)
    metal() do
        GC.@preserve A begin
            @atomic A[begin] += 1
        end
    end
    @test collect(A) == [2, 1, 1]
end


@testset "AtomixMetalExt:test_get_set" begin
    A = Metal.ones(Int32, 3)
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            x = Atomix.get(ref)
            Atomix.set!(ref, -x)
            A[2] = @atomic A[1]
            @atomic :monotonic A[3] = 2 * x
        end
    end
    @test collect(A) == [-1, -1, 2]
end


@testset "AtomixMetalExt:test_swap" begin
    A = Metal.MtlVector(Int32[1, 0, 0])
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            A[2] = Atomix.swap!(ref, Int32(5))
            A[3] = @atomicswap A[1] = Int32(7)
        end
    end
    @test collect(A) == [7, 1, 5]
end


@testset "AtomixMetalExt:test_ordering" begin
    A = Metal.ones(Int32, 2)
    metal() do
        GC.@preserve A begin
            @atomic :monotonic A[1] += 1
            @atomic :acquire_release A[2] -= 1
        end
    end
    @test collect(A) == [2, 0]
end


@testset "AtomixMetalExt:test_float" begin
    A = Metal.MtlVector(Float32[1, 1, 1, 1, 1, 0, 0, 1])
    metal() do
        GC.@preserve A begin
            @atomic A[1] += 1.5f0
            @atomic A[2] -= 0.5f0
            @atomic max(A[3], 3f0)
            @atomic min(A[4], -1f0)
            # no native instruction: compare-and-swap loop
            pre, post = @atomic A[5] * 4f0
            A[6] = pre + post
            A[7] = @atomicswap A[8] = 8f0
        end
    end
    @test collect(A) == [2.5, 0.5, 3, -1, 4, 5, 1, 8]
end
