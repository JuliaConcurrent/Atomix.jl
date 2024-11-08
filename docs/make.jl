using Documenter
using Atomix

makedocs(
    sitename = "Atomix",
    modules = [Atomix],
    warnonly = :missing_docs
)

deploydocs(
    repo = "github.com/JuliaConcurrent/Atomix.jl",
    devbranch = "main",
    push_preview = true,
)
