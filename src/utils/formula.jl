eachterm(@nospecialize(x::AbstractTerm)) = (x,)
eachterm(@nospecialize(x::NTuple{N,AbstractTerm})) where {N} = x

const ENDOG_CONTEXT = Any

struct EndogenousTerm{T} <: AbstractTerm
    x::T
end
StatsModels.termvars(t::EndogenousTerm) = [Symbol(t.x)]
endog(x::Symbol) = EndogenousTerm(term(x))
endog(x::Term) = EndogenousTerm(x)

has_endog(::EndogenousTerm) = true
has_endog(::FunctionTerm{typeof(endog)}) = true
has_endog(@nospecialize(t::InteractionTerm)) = any(has_endog(x) for x in t.terms)
has_endog(::AbstractTerm) = false
has_endog(f::Tuple) = any(has_endog(x) for x in f)
has_endog(@nospecialize(t::FormulaTerm)) = any(has_endog(x) for x in eachterm(t.lhs))

endogsymbol(t::EndogenousTerm) = Symbol(t.x)
endogsymbol(t::FunctionTerm{typeof(endog)}) = Symbol(t.args[1])

separate_slope_from_endog(t::FunctionTerm{typeof(endog)}) = (ConstantTerm(1), t)
separate_slope_from_endog(t::EndogenousTerm) = (ConstantTerm(1), t)

function separate_slope_from_endog(@nospecialize(t::InteractionTerm))
    slopeterms = filter(!has_endog, t.terms)
    endog_terms = filter(has_endog, t.terms)
    if length(endog_terms) > 1
        throw(ArgumentError("Interaction term contains more than one endogenous terms"))
        return slopeterms[1], endog_terms[1]
    end
    if length(slopeterms) > 1
        # throw(ArgumentError("Double-interaction with the endogenous term is not supported yet."))
        return InteractionTerm(slopeterms), endog_terms[1]
    else
        return slopeterms[1], endog_terms[1]
    end
end

has_categorical(t::AbstractTerm) = false
has_categorical(t::CategoricalTerm) = true
has_categorical(t::InteractionTerm) = any(has_categorical(x) for x in t.terms)

replace_function_term(@nospecialize(t::FunctionTerm{typeof(endog)})) = EndogenousTerm(t.args[1])
replace_function_term(t::AbstractTerm) = t
replace_function_term(@nospecialize(t::InteractionTerm)) =
    InteractionTerm(replace_function_term.(t.terms))
replace_function_term(@nospecialize(t::Tuple)) = replace_function_term.(t)
replace_function_term(@nospecialize(t::FormulaTerm)) =
    FormulaTerm(replace_function_term(t.lhs), replace_function_term(t.rhs))

function parse_endog(@nospecialize(f::FormulaTerm))
    if has_endog(f.rhs)
        throw(ArgumentError("Formula contains endogenous terms on the right-hand side"))
    end
    if has_endog(f.lhs)
        endog_terms = Tuple(term for term in eachterm(f.lhs) if has_endog(term))
        exog_terms = eachterm(f.rhs)
        response_terms = filter(!has_endog, eachterm(f.lhs))
        if length(response_terms) > 1
            throw(ArgumentError("Formula contains more than one response term"))
        end
        response_term = response_terms[1]
        separated_terms = [separate_slope_from_endog(term) for term in endog_terms]
        slope_terms = Tuple(term[1] for term in separated_terms)
        endog_terms = [term[2] for term in separated_terms]
        if length(unique(endogsymbol.(endog_terms))) .> 1
            throw(ArgumentError("Formula contains more than one endogenous term"))
        end
        endog_term = endog_terms[1]
        return response_term, slope_terms, endog_term, exog_terms
    else
        throw(ArgumentError("Formula does not contain endogenous terms"))
    end
end

function StatsModels.apply_schema(
    t::FunctionTerm{typeof(endog)},
    sch::StatsModels.Schema,
    Mod::Type{<:ENDOG_CONTEXT},
)
    return apply_schema(EndogenousTerm(t.args[1]), sch, Mod)
end

function StatsModels.apply_schema(
    t::EndogenousTerm,
    sch::StatsModels.Schema,
    Mod::Type{<:ENDOG_CONTEXT},
)
    return apply_schema(t.x, sch, Mod)
end

hint_for_categorical(df) = Dict{Symbol,Any}(Symbol(x) => CategoricalTerm for
                 x in names(df) if eltype(df[!, x]) <: Union{Int,Bool})
# ##=============== convert the formula to FixedEffectModel-compatible formula ================##
# function StatsModels.apply_schema(t::FunctionTerm{typeof(endog)}, sch::StatsModels.Schema, Mod::Type{<:_FEMODEL})
#     apply_schema(EndogenousTerm(t.args[1]), sch, Mod)
# end
# StatsModels.apply_schema(t::EndogenousTerm, schema, Mod::Type{<:_FEMODEL}) = apply_schema(t.x, schema, mod)

# function StatsModels.apply_schema(t::AbstractTerm, schema, Mod::Type{<:_ABSTRACTFEMODEL})
#     println("invoked")
#     t = apply_schema(t, schema, StatisticalModel)
#     if isa(t, CategoricalTerm)
#         t = FixedEffectModels.FixedEffectTerm(t.sym)
#     end
#     return t
# end

# function StatsModels.modelcols(e::EndogenousTerm, d::NamedTuple)
#     col = modelcols(p.)
# end
