using oneAPI
using oneAPI: @allowscalar


@testset "AtomixoneAPIExt:extension_found" begin
    @test !isnothing(Base.get_extension(Atomix, :AtomixoneAPIExt))
end


function oneapi(f)
    function g()
        f()
        nothing
    end
    oneAPI.@oneapi g()
end




@testset "AtomixoneAPIExt:test_cas" begin
    idx = (
        data = 1,
        cas1_ok = 2,
        cas2_ok = 3,
        # ...
    )
    @assert minimum(idx) >= 1
    @assert maximum(idx) == length(idx)

    A = oneAPI.zeros(Int32, length(idx))
    oneapi() do
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


@testset "AtomixoneAPIExt:test_inc" begin
    A = oneAPI.oneVector(Int32(1):Int32(3))
    oneapi() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            pre, post = Atomix.modify!(ref, +, 1)
            A[2] = pre
            A[3] = post
        end
    end
    @test collect(A) == [2, 1, 2]
end


@testset "AtomixoneAPIExt:test_inc_sugar" begin
    A = oneAPI.ones(Int32, 3)
    oneapi() do
        GC.@preserve A begin
            @atomic A[begin] += 1
        end
    end
    @test collect(A) == [2, 1, 1]
end


@testset "AtomixoneAPIExt:test_get_set" begin
    A = oneAPI.ones(Int32, 3)
    oneapi() do
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


@testset "AtomixoneAPIExt:test_swap" begin
    A = oneAPI.oneVector(Int32[1, 0, 0])
    oneapi() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            A[2] = Atomix.swap!(ref, Int32(5))
            A[3] = @atomicswap A[1] = Int32(7)
        end
    end
    @test collect(A) == [7, 1, 5]
end


@testset "AtomixoneAPIExt:test_ordering" begin
    A = oneAPI.ones(Int32, 2)
    oneapi() do
        GC.@preserve A begin
            @atomic :monotonic A[1] += 1
            @atomic :acquire_release A[2] -= 1
        end
    end
    @test collect(A) == [2, 0]
end


@testset "AtomixoneAPIExt:test_float" begin
    A = oneAPI.oneVector(Float32[1, 1, 1, 1, 1, 0])
    oneapi() do
        GC.@preserve A begin
            @atomic A[1] += 1.5f0
            @atomic A[2] -= 0.5f0
            @atomic max(A[3], 3f0)
            @atomic min(A[4], -1f0)
            # no native instruction: compare-and-swap loop
            pre, post = @atomic A[5] * 4f0
            A[6] = pre + post
        end
    end
    @test collect(A) == [2.5, 0.5, 3, -1, 4, 5]
end
