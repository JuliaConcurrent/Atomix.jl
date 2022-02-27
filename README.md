# AtomicArrays

AtomicArrays.jl have `@atomic`, `@atomicswap`, and `@atomicreplace` that are
superset of the macros in Base.jl.  In addition to atomic operations on the
fields, they support atomic operations on array elements.

```julia
julia> using AtomicArrays: @atomic, @atomicswap, @atomicreplace

julia> A = ones(Int, 3);

julia> @atomic A[1] += 1;  # fetch-and-increment

julia> @atomic A[1]
2

julia> @atomicreplace A[begin+1] 1 => 42  # compare-and-swap
(old = 1, success = true)

julia> @inbounds @atomic :monotonic A[begin+1]  # specify ordering and skip bound check
42

julia> @atomicswap A[end] = 123
1

julia> A[end]
123
```
