module SplitPolicy

using Flux

struct Policy
    model
    hidden_channels
    num_hidden_layers
    function Policy(in_channels, hidden_channels, num_hidden_layers)
        model = []
        push!(model, Dense(in_channels, hidden_channels, relu))
        for i in 1:num_hidden_layers-1
            push!(model, Dense(hidden_channels, hidden_channels, relu))
        end
        push!(model, Dense(hidden_channels, 2))
        model = Chain(model...)

        new(model, hidden_channels, num_hidden_layers)
    end
end

Flux.@functor Policy

function Base.show(io::IO, p::Policy)
    s = "Policy\n\t$(p.hidden_channels) channels\n\t$(p.num_hidden_layers) layers"
    println(io, s)
end

function (p::Policy)(state)
    return p.model(state)
end

end