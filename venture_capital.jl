# Yufei Xie, yux085@ucsd.edu
# Ewens, M., Gorbenko, A., & Korteweg, A. (2022). Venture capital contracts. Journal of Financial Economics, 143(1), 131-158.

using LinearAlgebra, Statistics, Suppressor
using Distributions, Expectations, Parameters, Plots, BenchmarkTools, DelimitedFiles

# -----------------------------------------------------------------------------------
#                                  1. Solving the model
# -----------------------------------------------------------------------------------


# Function Forms
g(i, e, ρ) = (0.5 * i^ρ + 0.5 * e^ρ)^(2/ρ)
h(c, β_1, β_2) = exp(β_1 * c + β_2 * c^2)
π_t(i, e, c, ρ, β_1, β_2) = g(i, e, ρ) * h(c, β_1, β_2)
α(c, γ_1) = 1- (1 - c) * exp(γ_1*(1 - c))
π_i(i, e, c, ρ, β_1, β_2, γ_1) = α(c, γ_1) * π_t(i, e, c, ρ, β_1, β_2)
π_e(i, e, c, ρ, β_1, β_2, γ_1) = (1 - α(c, γ_1)) * π_t(i, e, c, ρ, β_1, β_2)

# -----------------------------------------------------------------------------------

EGKModel = @with_kw (

    type_num = 60,
    i_vals = range(0.01, 10, length = type_num),
    e_vals = range(0.01, 10, length = type_num),

    max_iter = 2000, 
    tol = 5e-6,

    # contract, **initial value**
    c_vals = range(0, 1, length=200),

    # solutions
    c_star = zeros(type_num, type_num),
    π_i_star = zeros(type_num, type_num),
    π_e_star = zeros(type_num, type_num),
    π_t_star = zeros(type_num, type_num),
    M_star = zeros(type_num, type_num),
    
    ρ   = -1.370,
    β_1 = 0.679, 
    β_2 = -2.362,
    γ_1 = -0.211,

    λ_i = 13.443,
    λ_e = 10.393,
    r = 0.1,

    a_i = 1.927,
    b_i = 3.602,
    a_e = 3.142,
    b_e = 4.152,
    dist_i = BetaBinomial(type_num-1, a_i, b_i),
    dist_e = BetaBinomial(type_num-1, a_e, b_e),

    Wi_iv = collect(ones(length(i_vals))),
    Ve_iv = collect(ones(length(i_vals))),

    g = g, 
    h = h,
    α = α,
    π_i = π_i, 
    π_e = π_e,
    π_t = π_t)
egk = EGKModel()

function solve_optimal_contract(params)

    @unpack type_num, c_vals, i_vals, e_vals, c_star, π_i_star, π_e_star, π_t_star, M_star,
            β_1, β_2, γ_1, λ_i, λ_e, ρ, r, a_i, b_i, a_e, b_e, dist_i, dist_e, 
            Wi_iv, Ve_iv, g, h, α, π_i, π_e, π_t, max_iter, tol = params
    
    E_i = expectation(dist_i)
    E_e = expectation(dist_e)
    
    Wi_iv = π_i.(i_vals,i_vals, 0.45, ρ, β_1, β_2, γ_1)
    Ve_iv = π_e.(e_vals, e_vals, 0.45, ρ, β_1, β_2, γ_1)
    x_iv = [Wi_iv; Ve_iv]
    αi = α.(c_vals, γ_1)
    αe = 1 .- α.(c_vals, γ_1)
    h1 = h.(c_vals, β_1, β_2)

    # -------------------------------------------

    tmp_i = similar(collect(c_vals))
    tmp_e = copy(tmp_i)

    function T(x)
        Wi = x[1:type_num]
        Ve = x[type_num+1:end]
        for (i_num,i_type) in enumerate(i_vals)
            for (e_num,e_type) in enumerate(e_vals)

                tmp_i = αi .* h1 .* g(i_type, e_type, ρ)  # π_i
                tmp_e = αe .* h1 .* g(i_type, e_type, ρ) .- Ve[e_num]  # π_e - Ve
                (max_πi, max_id) = findmax(tmp_i .* (tmp_e ./ abs.(tmp_e) .+ abs.(tmp_e ./ abs.(tmp_e))) ./2)
                c_star[i_num,e_num] = c_vals[max_id]

                π_i_star[i_num,e_num] = max_πi #π_i(i_type, e_type, c_star[i_num,e_num], ρ, β_1, β_2, γ_1)
                π_e_star[i_num,e_num] = π_e(i_type, e_type, c_star[i_num,e_num], ρ, β_1, β_2, γ_1)

                if π_i_star[i_num,e_num] > Wi[i_num]
                    M_star[i_num,e_num] = 1
                else
                    M_star[i_num,e_num] = 0
                end
            end
        end

        [ (λ_i / (r + λ_i)) .* ((M_star .* π_i_star) * E_e.weights + ((1 .- M_star) * E_e.weights) .* Wi);
          (λ_e / (r + λ_e)) .* ((M_star .* π_e_star)' * E_i.weights + ((1 .- M_star)' * E_i.weights) .* Ve)]
        
    end
    
    # -------------------------------------------

    i = 0
    Error = 1
    v = copy(x_iv)
    v_next = similar(v)
    
    while i < max_iter && Error > tol
        v_next .= T(v)
        Error = norm(v_next - v)
        println("    $i iterations with error $Error")
        i += 1
        v .= v_next
        if  Error > 0.000005 && Error < 0.0000051 || i == max_iter - 20
            c_vals = range(0, 1, length = 4000)
            αi = α.(c_vals, γ_1)
            αe = 1 .- α.(c_vals, γ_1)
            h1 = h.(c_vals, β_1, β_2)
            tmp_i = similar(collect(c_vals))
            tmp_e = copy(tmp_i)
        else
        end

    end
    
    c_star = c_star .* M_star
    π_i_star = π_i_star .* M_star
    π_e_star = π_e_star .* M_star
    π_t_star = π_i_star .+ π_e_star
    Wi = Wi
    Ve = Ve

    return (c_star = c_star, M_star = M_star, π_i_star = π_i_star, π_e_star = π_e_star,
            π_t_star = π_t_star, Wi = Wi, Ve = Ve)

end

###################
####### 60x60 ≈ 50s 
###################
@time solve_optimal_contract(EGKModel(type_num = 60))


# ------------------------------------ 60x60 heatmap ---------------------------------

@unpack c_star, M_star, π_i_star, π_e_star, π_t_star, Wi, Ve = solve_optimal_contract(egk)
writedlm("0.0.1 c_star.csv", c_star, ",")
writedlm("0.0.2 M_star.csv", M_star, ",")
writedlm("0.0.3 π_i_star.csv", π_i_star, ",")
writedlm("0.0.4 π_e_star.csv", π_e_star, ",")
writedlm("0.0.5 π_t_star.csv", π_t_star, ",")

@unpack i_vals, e_vals = egk
heatmap(i_vals, e_vals, c_star', seriescolor=:binary, size=(450,400),
        title = "B. VC Equity Share", xlabel = "Investor", ylabel = "Entrepreneur",
        xtick = (0:2:10), ytick = (0:2:10))
Plots.savefig("0.1 VC equity share.svg")

heatmap(i_vals, e_vals, M_star', seriescolor=:binary, size=(450,400),
        title = "A. Matching", xlabel = "Investor", ylabel = "Entrepreneur",
        xtick = (0:2:10), ytick = (0:2:10))
plot!(i_vals, e_vals, color=:red, size=(450,400), linewidth = 1.5,
        label = "i = e", xtick = (0:2:10), ytick = (0:2:10))
Plots.savefig("0.2 Matching.svg")

heatmap(i_vals, e_vals, π_t_star', seriescolor=:Greys, size=(450,400),
        title = "C. Startup Value", xlabel = "Investor", ylabel = "Entrepreneur",
        xtick = (0:2:10), ytick = (0:2:10))
Plots.savefig("0.3 Startup Value.svg")

plot(i_vals, Wi, color=:red, size=(450,400), linewidth = 1.5, title = "D. Search Value", 
    xtick = (0:2:10), xlabel = "Investor/Entrepreneur", ylabel = "V(e) (W(i))", label= "Investor")
plot!(e_vals, Ve, color=:blue, size=(450,400), linewidth = 1.5, label="Entrepreneur")
Plots.savefig("0.4 Search Value.svg")





# -----------------------------------------------------------------------------------
#                                        2. Moments
# -----------------------------------------------------------------------------------

Moments = @with_kw (
    c_star = c_star,
    M_star = M_star,
    π_t_star = π_t_star)
mm = Moments()

function compute_moments(input)
    @unpack c_star, M_star, π_t_star = input
    @unpack dist_i, dist_e, type_num, λ_i = egk

    E_i = expectation(dist_i)
    E_e = expectation(dist_e)
    M_share = sum(M_star' .* (E_i.weights * E_e.weights'))
    M1 = sum(M_star[round(Int,type_num/4),:] .* E_e.weights)
    M2 = sum(M_star[round(Int,type_num/2),:] .* E_e.weights)
    M3 = sum(M_star[round(Int,3*type_num/4),:] .* E_e.weights)

    # average VC share of equity E(c*)
    E_c = sum(c_star' .* (E_i.weights * E_e.weights'))/ M_share

    # expected time to achieve a deal for an investor E(τ)
    p_i = M_star * E_e.weights
    τ_i = 1 ./ (p_i .* λ_i)
    M_τ = similar(M_star)
    for i = 1:length(τ_i)
        M_τ[i,:] .= τ_i[i]
    end
    E_τ = sum(M_star' .* M_τ' .* (E_i.weights * E_e.weights') )/ M_share

    # average firm value E(π(i,e,c*))
    E_π = sum(π_t_star' .* (E_i.weights * E_e.weights'))/ M_share

    return (E_c = E_c, E_τ = E_τ, E_π =E_π, M_share = M_share, M1 = M1, M2 = M2, M3 = M3)
end

moments = ones(1,7)
moments = compute_moments(mm)
writedlm("1 moments_main.csv", moments, ",")





# -----------------------------------------------------------------------------------
#                         3. Comparative Static: λi, λe, β1, β2 and γ
# -----------------------------------------------------------------------------------

l = 15

# ----------------------------------------- λi --------------------------------------

λ_i_vals = range(5, 20, length = l)
Res_λ_i = zeros(length(λ_i_vals), 7)
for (i, λ_i_val) in enumerate(λ_i_vals)
    
    egk = EGKModel(type_num = 40,λ_i = λ_i_val)
    @unpack c_star, M_star, π_i_star, π_e_star, π_t_star = solve_optimal_contract(egk)
    Moments = @with_kw (
        c_star = c_star,
        M_star = M_star,
        π_t_star = π_t_star)
    mm = Moments()
    @unpack (E_c, E_τ, E_π, M_share, M1, M2, M3) = compute_moments(mm)
    Res_λ_i[i, :] .= (E_c, E_τ, E_π, M_share, M1, M2, M3)
end

plot(λ_i_vals, Res_λ_i[:, 1], color=:black, size=(450,400), linewidth = 1.5,
    title = "A. VC Equity Share", xlabel = "λi", label = "Average VC Equity Share",
    xtick = (5:3:20))
Plots.savefig("2.1.1 λi.svg")

plot(λ_i_vals, Res_λ_i[:, 2], color=:black, size=(450,400), linewidth = 1.5,
    title = "B. Waiting Time (i)", xlabel = "λi", label = "Expected Waiting Time",
    xtick = (5:3:20))
Plots.savefig("2.1.2 λi.svg")

plot(λ_i_vals, Res_λ_i[:, 3], color=:black, size=(450,400), linewidth = 1.5,
    title = "C. Firm Value", xlabel = "λi", label = "Average Firm Value",
    xtick = (5:3:20))
Plots.savefig("2.1.3 λi.svg")

plot(λ_i_vals, Res_λ_i[:, 4], color=:black, size=(450,400), linewidth = 1.5,
    title = "D. Matching Probability", xlabel = "λi", label = "Market Size",
    xtick = (5:3:20))
Plots.savefig("2.1.4 λi.svg")

plot(λ_i_vals, Res_λ_i[:, 5], color=:blue, size=(450,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λi", label = "25%",
    xtick = (5:3:20))
plot!(λ_i_vals, Res_λ_i[:, 6], color=:red, size=(450,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λi", label = "50%",
    xtick = (5:3:20))
plot!(λ_i_vals, Res_λ_i[:, 7], color=:green, size=(450,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λi", label = "75%",
    xtick = (5:3:20))
Plots.savefig("2.1.5 λi.svg")

writedlm("2.1 λi.csv", Res_λ_i, ",")





# ----------------------------------------- λe --------------------------------------

λ_e_vals = range(5, 20, length = l)
Res_λ_e = zeros(length(λ_e_vals), 7)
for (i, λ_e_val) in enumerate(λ_e_vals)
    
    egk = EGKModel(type_num = 40, λ_e = λ_e_val)
    @unpack c_star, M_star, π_i_star, π_e_star, π_t_star = solve_optimal_contract(egk)
    Moments = @with_kw (
        c_star = c_star,
        M_star = M_star,
        π_t_star = π_t_star)
    mm = Moments()
    @unpack (E_c, E_τ, E_π, M_share, M1, M2, M3) = compute_moments(mm)
    Res_λ_e[i, :] .= (E_c, E_τ, E_π, M_share, M1, M2, M3)
end

plot(λ_e_vals, Res_λ_e[:, 1], color=:black, size=(450,400), linewidth = 1.5,
    title = " A. VC Equity Share", xlabel = "λe", label = "Average VC Equity Share",
    xtick = (5:3:20))
Plots.savefig("2.2.1 λe.svg")

plot(λ_e_vals, Res_λ_e[:, 2], color=:black, size=(450,400), linewidth = 1.5,
    title = "B. Waiting Time (i)", xlabel = "λe", label = "Expected Waiting Time (i)",
    xtick = (5:3:20))
Plots.savefig("2.2.2 λe.svg")

plot(λ_e_vals, Res_λ_e[:, 3], color=:black, size=(450,400), linewidth = 1.5,
    title = "C. Firm Value", xlabel = "λe", label = "Average Firm Value",
    xtick = (5:3:20))
Plots.savefig("2.2.3 λe.svg")

plot(λ_e_vals, Res_λ_e[:, 4], color=:black, size=(440,400), linewidth = 1.5,
    title = "D. Matching Probability", xlabel = "λe", label = "Market Size",
    xtick = (5:3:20))
Plots.savefig("2.2.4 λe.svg")

plot(λ_e_vals, Res_λ_e[:, 5], color=:blue, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λe", label = "25%",
    xtick = (5:3:20))
plot!(λ_e_vals, Res_λ_e[:, 6], color=:red, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λe", label = "50%",
    xtick = (5:3:20))
plot!(λ_e_vals, Res_λ_e[:, 7], color=:green, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "λe", label = "75%",
    xtick = (5:3:20))
Plots.savefig("2.2.5 λe.svg")

writedlm("2.2 λe.csv", Res_λ_e, ",")





# ----------------------------------------- β1 ---------------------------------------

β_1_vals = range(0.3, 1.5, length = l)
Res_β_1 = zeros(length(β_1_vals), 7)
for (i, β_1_val) in enumerate(β_1_vals)
    
    egk = EGKModel(type_num = 40, β_1 = β_1_val)
    @unpack c_star, M_star, π_i_star, π_e_star, π_t_star = solve_optimal_contract(egk)
    Moments = @with_kw (
        c_star = c_star,
        M_star = M_star,
        π_t_star = π_t_star)
    mm = Moments()
    @unpack (E_c, E_τ, E_π, M_share, M1, M2, M3) = compute_moments(mm)
    Res_β_1[i, :] .= (E_c, E_τ, E_π, M_share, M1, M2, M3)
end


plot(β_1_vals, Res_β_1[:, 1], color=:black, size=(450,400), linewidth = 1.5,
    title = " A. VC Equity Share", xlabel = "β1", label = "Average VC Equity Share",
    xtick = (0.3:0.3:1.5))
Plots.savefig("2.3.1 β1.svg")

plot(β_1_vals, Res_β_1[:, 2], color=:black, size=(450,400), linewidth = 1.5,
    title = "B. Waiting Time (i)", xlabel = "β1", label = "Expected Waiting Time (i)",
    xtick = (0.3:0.3:1.5))
Plots.savefig("2.3.2 β1.svg")

plot(β_1_vals, Res_β_1[:, 3], color=:black, size=(450,400), linewidth = 1.5,
    title = "C. Firm Value", xlabel = "β1", label = "Average Firm Value",
    xtick = (0.3:0.3:1.5))
Plots.savefig("2.3.3 β1.svg")

plot(β_1_vals, Res_β_1[:, 4], color=:black, size=(440,400), linewidth = 1.5,
    title = "D. Matching Probability", xlabel = "β1", label = "Market Size",
    xtick = (0.3:0.3:1.5))
Plots.savefig("2.3.4 β1.svg")

plot(β_1_vals, Res_β_1[:, 5], color=:blue, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β1", label = "25%",
    xtick = (0.3:0.3:1.5))
plot!(β_1_vals, Res_β_1[:, 6], color=:red, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β1", label = "50%",
    xtick = (0.3:0.3:1.5))
plot!(β_1_vals, Res_β_1[:, 7], color=:green, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β1", label = "75%",
    xtick = (0.3:0.3:1.5))
Plots.savefig("2.3.5 β1.svg")

writedlm("2.3 β1.csv", Res_β_1, ",")





# ----------------------------------------- β2 ---------------------------------------

β_2_vals = range(-5, -1.1, length = l)
Res_β_2 = zeros(length(β_2_vals), 7)
for (i, β_2_val) in enumerate(β_2_vals)
    
    egk = EGKModel(type_num = 40, β_2 = β_2_val)
    @unpack c_star, M_star, π_i_star, π_e_star, π_t_star = solve_optimal_contract(egk)
    Moments = @with_kw (
        c_star = c_star,
        M_star = M_star,
        π_t_star = π_t_star)
    mm = Moments()
    @unpack (E_c, E_τ, E_π, M_share, M1, M2, M3) = compute_moments(mm)
    Res_β_2[i, :] .= (E_c, E_τ, E_π, M_share, M1, M2, M3)
end


plot(β_2_vals, Res_β_2[:, 1], color=:black, size=(450,400), linewidth = 1.5,
    title = " A. VC Equity Share", xlabel = "β2", label = "Average VC Equity Share",
    xtick = (-5:1.3:-1.1))
Plots.savefig("2.4.1 β2.svg")

plot(β_2_vals, Res_β_2[:, 2], color=:black, size=(450,400), linewidth = 1.5,
    title = "B. Waiting Time (i)", xlabel = "β2", label = "Expected Waiting Time (i)",
    xtick = (-5:1.3:-1.1))
Plots.savefig("2.4.2 β2.svg")

plot(β_2_vals, Res_β_2[:, 3], color=:black, size=(450,400), linewidth = 1.5,
    title = "C. Firm Value", xlabel = "β2", label = "Average Firm Value",
    xtick = (-5:1.3:-1.1))
Plots.savefig("2.4.3 β2.svg")

plot(β_2_vals, Res_β_2[:, 4], color=:black, size=(440,400), linewidth = 1.5,
    title = "D. Matching Probability", xlabel = "β2", label = "Market Size",
    xtick = (-5:1.3:-1.1))
Plots.savefig("2.4.4 β2.svg")

plot(β_2_vals, Res_β_2[:, 5], color=:blue, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β2", label = "25%",
    xtick = (-5:1.3:-1.1))
plot!(β_2_vals, Res_β_2[:, 6], color=:red, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β2", label = "50%",
    xtick = (-5:1.3:-1.1))
plot!(β_2_vals, Res_β_2[:, 7], color=:green, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "β2", label = "75%",
    xtick = (-5:1.3:-1.1))
Plots.savefig("2.4.5 β2.svg")

writedlm("2.4 β2.csv", Res_β_2, ",")





# ----------------------------------------- γ1 ---------------------------------------

γ_1_vals = range(-1, 0, length = l)
Res_γ_1 = zeros(length(γ_1_vals), 7)
for (i, γ_1_val) in enumerate(γ_1_vals)
    
    egk = EGKModel(type_num = 40, γ_1 = γ_1_val)
    @unpack c_star, M_star, π_i_star, π_e_star, π_t_star = solve_optimal_contract(egk)
    Moments = @with_kw (
        c_star = c_star,
        M_star = M_star,
        π_t_star = π_t_star)
    mm = Moments()
    @unpack (E_c, E_τ, E_π, M_share, M1, M2, M3) = compute_moments(mm)
    Res_γ_1[i, :] .= (E_c, E_τ, E_π, M_share, M1, M2, M3)
end


plot(γ_1_vals, Res_γ_1[:, 1], color=:black, size=(450,400), linewidth = 1.5,
    title = " A. VC Equity Share", xlabel = "γ1", label = "Average VC Equity Share",
    xtick = (-1:0.2:0))
Plots.savefig("2.5.1 γ1.svg")

plot(γ_1_vals, Res_γ_1[:, 2], color=:black, size=(450,400), linewidth = 1.5,
    title = "B. Waiting Time (i)", xlabel = "γ1", label = "Expected Waiting Time (i)",
    xtick = (-1:0.2:0))
Plots.savefig("2.5.2 γ1.svg")

plot(γ_1_vals, Res_γ_1[:, 3], color=:black, size=(450,400), linewidth = 1.5,
    title = "C. Firm Value", xlabel = "γ1", label = "Average Firm Value",
    xtick = (-1:0.2:0))
Plots.savefig("2.5.3 γ1.svg")

plot(γ_1_vals, Res_γ_1[:, 4], color=:black, size=(440,400), linewidth = 1.5,
    title = "D. Matching Probability", xlabel = "γ1", label = "Market Size",
    xtick = (-1:0.2:0))
Plots.savefig("2.5.4 γ1.svg")

plot(γ_1_vals, Res_γ_1[:, 5], color=:blue, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "γ1", label = "25%",
    xtick = (-1:0.2:0))
plot!(γ_1_vals, Res_γ_1[:, 6], color=:red, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "γ1", label = "50%",
    xtick = (-1:0.2:0))
plot!(γ_1_vals, Res_γ_1[:, 7], color=:green, size=(440,400), linewidth = 1.5,
    title = "E. Investor", xlabel = "γ1", label = "75%",
    xtick = (-1:0.2:0))
Plots.savefig("2.5.5 γ1.svg")

writedlm("2.5 γ1.csv", Res_γ_1, ",")








# h(c, β1)
c = range(0, 1, length = 1000)
h(c, β_1, β_2) = exp(β_1 * c + β_2 * c^2)
y1 = h.(c, 0.3, -2.362)
y2 = h.(c, 1, -2.362)
plot(c, y1, color=:blue, size=(450,400), linewidth = 1.5,
    title = "h(c)", xlabel = "c", label = "β1 = 0.3", xtick = (0:0.2:1))
plot!(c, y2, color=:red, size=(450,400), linewidth = 1.5,
    title = "h(c)", xlabel = "c", ylabel = "h(c)", label = "β1 = 1", xtick = (0:0.2:1))
Plots.savefig("2.3 h(c)_β1.svg")
# h(c, β2)
z1 = h.(c, 0.679, -4)
z2 = h.(c, 0.679, -1)
plot(c, z1, color=:blue, size=(450,400), linewidth = 1.5,
    title = "h(c)", xlabel = "c", label = "β2 = -4", xtick = (0:0.2:1))
plot!(c, z2, color=:red, size=(450,400), linewidth = 1.5,
    title = "h(c)", xlabel = "c", ylabel = "h(c)", label = "β2 = -1", xtick = (0:0.2:1))
Plots.savefig("2.3 h(c)_β2.svg")
# α(c, γ1)
α(c, γ_1) = 1- (1 - c) * exp(γ_1*(1 - c))
x1 = α.(c, -1)
x2 = α.(c, -0.5)
x3 = α.(c, 0)
plot(c, x1, color=:blue, size=(450,400), linewidth = 1.5,
    title = "α(c)", xlabel = "c", label = "γ1 = -1", xtick = (0:0.2:1))
plot!(c, x2, color=:red, size=(450,400), linewidth = 1.5,
    title = "α(c)", xlabel = "c", label = "γ1 = -0.5", xtick = (0:0.2:1))
plot!(c, x3, color=:green, size=(450,400), linewidth = 1.5,
    title = "α(c)", xlabel = "c", label = "γ1 = 0", xtick = (0:0.2:1))
Plots.savefig("2.5 α(c).svg")
