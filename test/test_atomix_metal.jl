using Metal
using Metal: @allowscalar
using UnsafeAtomics: UnsafeAtomics


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


function compile_metal_4_1(f)
    function g()
        f()
        nothing
    end
    return sprint() do io
        Metal.code_llvm(
            io,
            g,
            Tuple{};
            kernel=true,
            metal=v"4.1",
            air=v"2.9",
            dump_module=true,
        )
    end
end


if Metal.metal_target() >= v"4.1"

@testset "AtomixMetalExt:test_get_set" begin
    A = Metal.ones(Int32, 3)
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            Atomix.set!(ref, -Atomix.get(ref, Atomix.acquire), Atomix.release)
            A[2] = Atomix.get(ref, Atomix.acquire)
            Atomix.set!(ref, Int32(7), UnsafeAtomics.unordered)
            A[3] = Atomix.get(ref, Atomix.monotonic)
        end
    end
    @test collect(A) == Int32[7, -1, 7]
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


@testset "AtomixMetalExt:test_orderings" begin
    A = Metal.zeros(Int32, 4)
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            Atomix.modify!(ref, +, 1, UnsafeAtomics.unordered)
            Atomix.modify!(ref, +, 1, Atomix.monotonic)
            Atomix.modify!(ref, +, 1, Atomix.acquire)
            Atomix.modify!(ref, +, 1, Atomix.release)
            Atomix.modify!(ref, +, 1, Atomix.acquire_release)
            old, success = Atomix.replace!(
                ref,
                5,
                10,
                Atomix.sequentially_consistent,
                Atomix.acquire,
            )
            A[2] = old
            A[3] = success
            old, success = Atomix.replace!(
                ref,
                5,
                11,
                Atomix.acq_rel,
                Atomix.sequentially_consistent,
            )
            A[4] = old + success
        end
    end
    @test collect(A) == Int32[10, 5, 1, 10]
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

else

@testset "AtomixMetalExt:ordered_atomics_compile" begin
    A = Metal.zeros(Int32, 1)
    llvm = compile_metal_4_1() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            value = Atomix.get(ref, Atomix.acquire)
            Atomix.set!(ref, value, Atomix.release)
            Atomix.modify!(ref, +, 1, Atomix.acquire_release)
            Atomix.replace!(
                ref,
                1,
                2,
                Atomix.sequentially_consistent,
                Atomix.acquire,
            )
        end
    end
    @test occursin("air.atomic", llvm)
end

@testset "AtomixMetalExt:ordered_atomics_runtime" begin
    @test_skip Metal.metal_target() >= v"4.1"
end

end
