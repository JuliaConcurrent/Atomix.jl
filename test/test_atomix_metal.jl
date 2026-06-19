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


# Not implemented:
#=
function test_get_set()
    A = CUDA.ones(Int, 3)
    cuda() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            x = Atomix.get(ref)
            Atomix.set!(ref, -x)
        end
    end
    @test collect(A) == [-1, 1, 1]
end
=#


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


@testset "AtomixMetalExt:test_complex_cas" begin
    A = Metal.zeros(ComplexF64, 3)
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            (old, success) = Atomix.replace!(ref, ComplexF64(0.0, 0.0), ComplexF64(1.0, 2.0))
            A[2] = old
            A[3] = success ? ComplexF64(1.0) : ComplexF64(0.0)
        end
    end
    result = collect(A)
    @test result[1] == ComplexF64(1.0, 2.0)
    @test result[2] == ComplexF64(0.0, 0.0)
    @test result[3] == ComplexF64(1.0)
end


@testset "AtomixMetalExt:test_complex_modify" begin
    A = Metal.fill(ComplexF64(1.0, 2.0), 3)
    metal() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            pre, post = Atomix.modify!(ref, +, ComplexF64(3.0, 4.0))
            A[2] = pre
            A[3] = post
        end
    end
    result = collect(A)
    @test result[1] == ComplexF64(4.0, 6.0)
    @test result[2] == ComplexF64(1.0, 2.0)
    @test result[3] == ComplexF64(4.0, 6.0)
end


@testset "AtomixMetalExt:test_complex_sugar" begin
    A = Metal.ones(ComplexF64, 3)
    metal() do
        GC.@preserve A begin
            @atomic A[begin] += ComplexF64(2.0, 3.0)
        end
    end
    result = collect(A)
    @test result[1] == ComplexF64(3.0, 3.0)
    @test result[2] == ComplexF64(1.0, 0.0)
    @test result[3] == ComplexF64(1.0, 0.0)
end
