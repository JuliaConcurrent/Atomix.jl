macro atomic(ex)
    ans = handle_array(QuoteNode(:sequentially_consistent), ex)
    ans === nothing || return ans
    esc(:($Base.@atomic($ex)))
end

macro atomic(order, ex)
    ans = handle_array(order, ex)
    ans === nothing || return ans
    esc(:($Base.@atomic($order, $ex)))
end

macro atomic(a1, op, a2)
    ans = handle_array(QuoteNode(:sequentially_consistent), a1, op, a2)
    ans === nothing || return ans
    esc(:($Base.@atomic($a1, $op, $a2)))
end

macro atomic(order, a1, op, a2)
    ans = handle_array(order, a1, op, a2)
    ans === nothing || return ans
    esc(:($Base.@atomic($order, $a1, $op, $a2)))
end

function asref_expr(ex)
    @nospecialize
    isexpr(ex, :ref) || return nothing
    array = esc(ex.args[1])
    indices = map(esc, ex.args[2:end])
    return :(asref($array)[$(indices...)])
end

function order_expr(order)
    @nospecialize
    if order isa QuoteNode
        llvm_ordering_from_juila(order.value)
    else
        :(llvm_ordering_from_juila($(esc(order))))
    end
end

function handle_array(order, ex)
    @nospecialize
    if ex isa Expr
        if (ref = asref_expr(ex)) !== nothing
            return :(UnsafeAtomics.load($ref, $(order_expr(order))))
        elseif isexpr(ex, :call, 3)
            return handle_array(order, ex.args[2], ex.args[1], ex.args[3])
        elseif ex.head === :(=)
            l, r = ex.args[1], esc(ex.args[2])
            if (ref = asref_expr(l)) !== nothing
                return :(UnsafeAtomics.store!($ref, $r, $(order_expr(order))))
            end
        elseif length(ex.args) == 2
            shead = string(ex.head)
            if endswith(shead, '=')
                op = Symbol(shead[1:prevind(shead, end)])
                ans = handle_array(order, ex.args[1], op, ex.args[2])
                ans === nothing || return :($ans[2])
            end
        end
    end
    return nothing
end

function handle_array(order, a1, op, a2)
    @nospecialize
    ref = asref_expr(a1)
    ref === nothing && return nothing
    :(UnsafeAtomics.modify!($ref, $(esc(op)), $(esc(a2)), $(order_expr(order))))
end
