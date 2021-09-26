# AtomicArrays

```julia
julia> using AtomicArrays

julia> A = AtomicRefArray(zeros(Int, 3));

julia> ref = A[1];

julia> ref[]
0
```

The value of a reference `A[i]` can be updated through the property `._` and
`@atomic*` API provided by Julia ≥ 1.7.

```julia
julia> @atomic A[1]._
0

julia> @atomic A[1]._ += 1
1

julia> @atomicreplace A[1]._ 1 => 2
(old = 1, success = true)

julia> @atomicswap A[1]._ = 3
2

julia> @atomic A[1]._
3
```
