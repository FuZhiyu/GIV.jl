using Test, GIV, Random
using DataFrames, CSV
Random.seed!(6)
simmodel = GIV.SimModel(T = 400, N = 100, varᵤshare = 0.8, usupplyshare = 0.0, h = 0.3, σᵤcurv = 0.2, ζs = 0.0, NC = 2, M = 0.5, σζ = 0.0)
df = DataFrame(simmodel.data)
df.group = mod.(df.id , 10)

f =@formula(q + group &endog(p) ~ id & (η1 + η2))
id = :id
t = :t
weight= :absS
df = preprocess_dataframe(df, f, :id, :t, :absS; quiet = true)
@time givmodel_uu = giv(
    df,
    f,
    :id,
    :t,
    :absS;
    guess = Dict("group" => ones(10)),
    algorithm = :uu,
)

@time givmodel_uu = giv(
    df,
    f,
    :id,
    :t,
    :absS;
    guess = Dict("group" => ones(10)),
    algorithm = :vcov,
)

# qmat, pmat, Cts, ηts, Smat, uqmat, λq, uCpts, λCp, meanqmat, meanpmat, meanCpts, meanηts = GIV.generate_matrices(df, f, id, t, weight; algorithm = :uu, quiet = false)

# @time Cqq, CqCp, CCpq, CCpCp, qq, Cpq, CpCp = GIV.compuate_covariance_tensors(uqmat, uCpts, Cts)
