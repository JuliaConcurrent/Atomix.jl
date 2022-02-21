baremodule AtomicArrays

export AtomicRefArray

import Base

macro atomic end

struct AtomicRef{T,Ptr,Data}
    eltype::Base.Val{T}
    ptr::Ptr
    data::Data
end

struct AtomicRefArray{T,N,Data<:Base.AbstractArray{T,N}} <:
       Base.AbstractArray{AtomicRef{T,Data},N}
    data::Data
end

function asref end

module Internal

import ..AtomicArrays: AtomicRef, AtomicRefArray, @atomic
using ..AtomicArrays: AtomicArrays, asref

using Base.Meta: isexpr
using Base: @propagate_inbounds
using UnsafeAtomics:
    Ordering, UnsafeAtomics, monotonic, acquire, release, acq_rel, seq_cst, right

include("utils.jl")
include("core.jl")
include("sugar.jl")
if isdefined(Base, :replaceproperty!)  # 1.7 or later
    include("properties.jl")
end

function define_docstring()
    path = joinpath(@__DIR__, "..", "README.md")
    include_dependency(path)
    doc = read(path, String)
    doc = replace(doc, r"^```julia"m => "```jldoctest README")
    # Setting `LineNumberNode` to workaround an error from logging(?) `no method
    # matching getindex(::Nothing, ::Int64)`:
    ex = :($Base.@doc $doc AtomicArrays)
    ex.args[2]::LineNumberNode
    ex.args[2] = LineNumberNode(1, Symbol(path))
    Base.eval(AtomicArrays, ex)
end

end  # module Internal

Internal.define_docstring()

end  # baremodule AtomicArrays
