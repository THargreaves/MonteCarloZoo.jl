using Documenter, MonteCarloZoo

push!(LOAD_PATH,"../src/")
makedocs(sitename="MonteCarloZoo.jl Documentation",
         pages = [
            "Index" => "index.md",
         ],
         format = Documenter.HTML(prettyurls = false)
)

deploydocs(
    repo = "github.com/THargreaves/MonteCarloZoo.jl.git",
    devbranch = "master"
)
