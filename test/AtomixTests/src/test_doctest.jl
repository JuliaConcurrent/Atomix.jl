module TestDoctest

import AtomicArrays
using Documenter: doctest
using Test

function test_doctest()
    doctest(AtomicArrays; manual = false)
end

end  # module
