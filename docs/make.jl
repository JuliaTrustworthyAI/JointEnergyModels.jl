using JointEnergyModels
using Documenter

include("setup_docs.jl")

DocMeta.setdocmeta!(JointEnergyModels, :DocTestSetup, :(setup_docs); recursive=true)

makedocs(;
    modules=[JointEnergyModels],
    authors="Patrick Altmeyer",
    repo="https://github.com/JuliaTrustworthyAI/JointEnergyModels.jl/blob/{commit}{path}#{line}",
    sitename="JointEnergyModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://juliatrustworthyai.github.io/JointEnergyModels.jl",
        edit_link="main",
        assets=String[]
    ),
    pages=[
        "🏠 Home" => "index.md",
        # "🫣 Tutorials" => [
        #     "Overview" => "tutorials/index.md",
        # ],
        # "🤓 Explanation" => [
        #     "Overview" => "explanation/index.md",
        # ],
        # "🫡 How-To ..." => [
        #     "Overview" => "how_to_guides/index.md",
        # ],
        # "🧐 Reference" => "reference.md",
        # "🛠 Contribute" => "contribute.md",
        # "📚 Additional Resources" => "assets/resources.md",
    ]
)

deploydocs(;
    repo="github.com/JuliaTrustworthyAI/JointEnergyModels.jl",
    devbranch="main",
)
