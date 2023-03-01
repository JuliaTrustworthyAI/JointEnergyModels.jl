using JointEnergyModels
using Documenter

DocMeta.setdocmeta!(JointEnergyModels, :DocTestSetup, :(using JointEnergyModels); recursive=true)

makedocs(;
    modules=[JointEnergyModels],
    authors="Patrick Altmeyer",
    repo="https://github.com/pat-alt/JointEnergyModels.jl/blob/{commit}{path}#{line}",
    sitename="JointEnergyModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://pat-alt.github.io/JointEnergyModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/pat-alt/JointEnergyModels.jl",
    devbranch="main",
)
