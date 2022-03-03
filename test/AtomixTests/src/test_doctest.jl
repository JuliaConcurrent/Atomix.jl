module TestDoctest

import Atomix
using Documenter: doctest
using Test

function test_doctest()
    doctest(Atomix; manual = false)
end

end  # module
