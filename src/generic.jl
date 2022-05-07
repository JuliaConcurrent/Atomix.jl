# TODO: Support Symbol ordering

@inline Atomix.get(m, lens) = Atomix.get(m, lens, seq_cst)
@inline Atomix.set!(m, lens, x) = Atomix.set!(m, lens, x, seq_cst)

@inline Atomix.modify!(m, lens, op::OP, x) where {OP} =
    Atomix.modify!(m, lens, op, x, seq_cst)

@inline Atomix.replace!(m, lens, expected, desired, order::Ordering = seq_cst) =
    Atomix.replace!(m, lens, expected, desired, order, order)

@inline Atomix.swap!(m, lens, x, order::Ordering = seq_cst) =
    first(Atomix.modify!(m, lens, op, x, order))
