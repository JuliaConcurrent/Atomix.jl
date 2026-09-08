using OpenCL, pocl_jll
using OpenCL: @allowscalar
cl.platform!("pocl")

@testset "AtomixOpenCLExt:extension_found" begin
    @test !isnothing(Base.get_extension(Atomix, :AtomixOpenCLExt))
end


function opencl(f)
    function g()
        f()
        nothing
    end
    OpenCL.@opencl g()
end




@testset "AtomixOpenCLExt:test_cas" begin
    idx = (
        data = 1,
        cas1_ok = 2,
        cas2_ok = 3,
        # ...
    )
    @assert minimum(idx) >= 1
    @assert maximum(idx) == length(idx)

    A = OpenCL.zeros(Int32, length(idx))
    opencl() do
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


@testset "AtomixOpenCLExt:test_inc" begin
    A = OpenCL.CLArray(Int32(1):Int32(3))
    opencl() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            pre, post = Atomix.modify!(ref, +, 1)
            A[2] = pre
            A[3] = post
        end
    end
    @test collect(A) == [2, 1, 2]
end


@testset "AtomixOpenCLExt:test_inc_sugar" begin
    A = OpenCL.ones(Int32, 3)
    opencl() do
        GC.@preserve A begin
            @atomic A[begin] += 1
        end
    end
    @test collect(A) == [2, 1, 1]
end


@testset "AtomixOpenCLExt:test_get_set" begin
    A = OpenCL.ones(Int32, 3)
    opencl() do
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


@testset "AtomixOpenCLExt:test_swap" begin
    A = OpenCL.CLArray(Int32[1, 0, 0])
    opencl() do
        GC.@preserve A begin
            ref = Atomix.IndexableRef(A, (1,))
            A[2] = Atomix.swap!(ref, Int32(5))
            A[3] = @atomicswap A[1] = Int32(7)
        end
    end
    @test collect(A) == [7, 1, 5]
end


@testset "AtomixOpenCLExt:test_ordering" begin
    A = OpenCL.ones(Int32, 2)
    opencl() do
        GC.@preserve A begin
            @atomic :monotonic A[1] += 1
            @atomic :acquire_release A[2] -= 1
        end
    end
    @test collect(A) == [2, 0]
end


@testset "AtomixOpenCLExt:test_float" begin
    A = OpenCL.CLArray(Float32[1, 1, 1, 1, 1, 0, 0, 1])
    opencl() do
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


@testset "AtomixOpenCLExt:test_float64" begin
    A = OpenCL.CLArray(Float64[1, 1, 1, 1, 1, 0, 0, 1])
    opencl() do
        GC.@preserve A begin
            @atomic A[1] += 1.5
            @atomic A[2] -= 0.5
            @atomic max(A[3], 3.0)
            @atomic min(A[4], -1.0)
            pre, post = @atomic A[5] * 4.0
            A[6] = pre + post
            A[7] = @atomicswap A[8] = 8.0
        end
    end
    @test collect(A) == [2.5, 0.5, 3, -1, 4, 5, 1, 8]
end
