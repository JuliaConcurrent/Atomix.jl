@noinline nonatomic_field_error(name::Symbol) =
    ConcurrencyViolationError("property $name cannot be accessed atomically")

@inline invalid_ordering(name::Symbol) =
    ConcurrencyViolationError("invalid ordering: $name")

_JULIA_ORDERINGS =
    [:unordered, :monotonic, :acquire, :release, :acquire_release, :sequentially_consistent]

let body = foldr(_JULIA_ORDERINGS, init = :(invalid_ordering(order))) do name, ex
        quote
            if order === $(QuoteNode(name))
                f(UnsafeAtomics.$name)
            else
                $ex
            end
        end
    end

    @eval @inline function with_llvm_order_from_julia_order(f, order::Symbol)
        $body
    end
end

@inline function Base.getproperty(ref::AtomicRef, name::Symbol, order::Symbol)
    if name === :_
        @inline _get(order::Ordering) = UnsafeAtomics.load(ref, order)
        return with_llvm_order_from_julia_order(_get, order)
    end
    nonatomic_field_error(name)
end

@inline function Base.setproperty!(ref::AtomicRef, name::Symbol, x, order::Symbol)
    if name === :_
        @inline _set!(order::Ordering) = UnsafeAtomics.store!(ref, x, order)
        return with_llvm_order_from_julia_order(_set!, order)
    end
    nonatomic_field_error(name)
end

let body = foldr(_JULIA_ORDERINGS, init = :(invalid_ordering(order))) do succ, ex1
        ex2 = foldr(_JULIA_ORDERINGS, init = :(invalid_ordering(order))) do fail, ex
            quote
                if fail_order === $(QuoteNode(fail))
                    return UnsafeAtomics.cas!(
                        ref,
                        expected,
                        desired,
                        UnsafeAtomics.$succ,
                        UnsafeAtomics.$fail,
                    )
                else
                    $ex
                end
            end
        end
        quote
            if success_order === $(QuoteNode(succ))
                $ex2
            else
                $ex1
            end
        end
    end

    @eval @inline function Base.replaceproperty!(
        ref::AtomicRef,
        name::Symbol,
        expected,
        desired,
        success_order::Symbol,
        fail_order::Symbol,
    )
        if name === :_
            $body
        end
        nonatomic_field_error(name)
    end
end

@inline function Base.modifyproperty!(
    ref::AtomicRef,
    name::Symbol,
    op::OP,
    x,
    order::Symbol,
) where {OP}
    if name === :_
        @inline _modify!(order::Ordering) = UnsafeAtomics.modify!(ref, op, x, order)
        return with_llvm_order_from_julia_order(_modify!, order)
    end
    nonatomic_field_error(name)
end

@inline Base.swapproperty!(ref::AtomicRef, name::Symbol, x, order::Symbol) =
    first(modifyproperty!(ref, name, right, x, order))
