function predict(p, IC, tspan, saveat = t_obs)  
    _prob = ODEProblem{true}(dynamics!, IC, tspan, p)
    Array(solve(_prob, Tsit5(), saveat = saveat))
end

# Convert the vector of parameters to a named tuple.
function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end