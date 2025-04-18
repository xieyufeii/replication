# Yufei Xie, yux085@ucsd.edu

using Plots, Parameters, SparseArrays, LinearAlgebra, Statistics, Roots
using Distributions, Expectations, Plots, BenchmarkTools, DelimitedFiles

# utility
function u(c::Float64, ρ::Float64)
    if c<=0
        u = -Inf64;
    else
        if ρ != 1
            u = (c^(1-ρ))/(1-ρ)
        else
            u = log(c)
        end
    end
    return u
end

# tauchen
function tauchen(N::Int64, μ::Float64, ρ::Float64, σ::Float64, m::Int64)
   	zN      = m*sqrt(σ^2/(1-ρ^2))
    z1      = -zN
    a       = (1-ρ)*μ
    z       = collect(range(z1, zN, length = N))
    z       = z .+ a/(1-ρ)
    step    = (zN-z1)/(N-1)  # 2*m*sqrt(σ^2/(1-ρ^2))/(N-1)
    zprob   = fill(0.0, N, N)
    for i = 1:N
        for j = 1:N
            if j == 1
                zprob[i,j] = cdf(Normal(), (z[1] + step/2.0 - a - ρ*z[i])/σ)
            elseif j == N
                zprob[i,j] = 1 - cdf(Normal(), (z[N] - step/2.0 - a - ρ*z[i])/σ)
            else
                zprob[i,j] = cdf(Normal(), (z[j] + step/2.0 - a - ρ*z[i])/σ) -
                             cdf(Normal(), (z[j] - step/2.0 - a - ρ*z[i])/σ)
            end
        end
	end
    ps = sum(zprob, dims = 2)
    zprob = zprob./ps
    return z, zprob
end

# parameters
@with_kw struct param
    β::Float64 = 0.96; 
    ρ::Float64 = 2;

    Z::Int64 = 2;
    μ_z::Float64 = 1.0;
    ρ_z::Float64 = 0.6; # ρ ∈ {0.3, 0.6, 0.9}
    σ_z::Float64 = 0.2; # σ ∈ {0.2, 0.4}
    m_z::Int64 = 3;
    A::Float64 = 1.0; # tfp

    b::Float64 = 0.0;
    K::Int64 = 200;
    kmin::Float64 = -b;
    kmax::Float64 = 10;
    k_vec = collect(range(kmin,kmax,K));

    δ::Float64 = 0.1;
    α::Float64 = 0.33;

    max_iter_v::Int64 = 1000;
    max_iter_λ::Int64 = 1000;
    max_iter_k::Int64 = 30;
    tol_v::Float64 = 1e-5;
    tol_λ::Float64 = 1e-10;
    tol_k::Float64 = 1e-4;
    upda::Float64 = 0.7;
end
p = param();

function solve_eq(K_guess, param)

    @unpack β,ρ,K,Z,μ_z,ρ_z,σ_z,m_z,A,k_vec,δ,α,max_iter_v,max_iter_λ,tol_v,tol_λ,upda = param

    # z_vec, P_z = tauchen(Z, μ_z, ρ_z, σ_z, m_z); # more z states
    z_vec = [0.3 1.0]';
    P_z = [0.75 0.25; 0.75 0.25];
    z_sta = P_z^10000;
    z_sta = z_sta[1,:];
    L = sum(z_vec'*z_sta); # total labor supply

    # initial guess
    v = ones(Z,K);
    λ = ones(Z,K)./sum(ones(Z,K));
    a_prime = zeros(Z,K);
    v_tmp = zeros(Z,K,K);
    c_tmp = zeros(Z,K,K);
    u_tmp = zeros(Z,K,K);

    # prices
    w = (1-α)*A*K_guess^(α)*L^(-α)
    r = α*A*K_guess^(α-1)*L^(1-α) - δ

    # consumption
    c_tmp = w.*reshape(z_vec,Z,1,1) .+ (1+r).*reshape(k_vec,1,K,1) .- reshape(k_vec,1,1,K);
    u_tmp = u.(c_tmp, ρ);

    # vfi
    function Tv(v)
        Ev = P_z*v;
        v_tmp = u_tmp .+ β*reshape(Ev,Z,1,K);
        v_prime, a_prime = findmax(v_tmp, dims=3)
        v_prime = dropdims(v_prime, dims=3)
        a_prime = dropdims(a_prime, dims=3) .|> x -> x[3]
        return a_prime, v_prime
    end

    v_next = similar(v)
    i = 0;
    Error = 1;
    while i < max_iter_v && Error > tol_v
        a_next, v_next = Tv(v)
        Error = norm(v_next .- v)
        # if i % 10 == 0
        #     println("v:    $i iterations with error $Error")
        # end
        v .= (1-upda)*v + upda*v_next
        a_prime = a_next
        i = i+1;
    end

    # stationary distribution
    function Tλ(λ)
        S = reshape(sparse(1:Z*K,vec(a_prime),vec(λ),Z*K,K),Z,K,K)
        λ_tmp = reshape(sum(S,dims=2),Z,K)
        λ_prime = P_z'*λ_tmp
        return λ_prime
    end

    λ_next = similar(λ);
    i = 0;
    Error = 1;
    while i < max_iter_λ && Error > tol_λ
        λ_next = Tλ(λ)
        Error = norm(λ_next .- λ)
        # if i % 10 == 0
        #     println("λ:    $i iterations with error $Error")
        # end
        λ .= (1-upda)*λ + upda*λ_next
        i = i+1;
    end

    # agg capital supply
    k_supply = sum((zeros(Z,1).+k_vec').*λ)

    return k_supply, λ, v, a_prime, r, w
end

function find_eq_k(k_guess, param)
    k_supply, _, _, _, _, _ = solve_eq(k_guess, param)
    println("Iterating with capital = ", k_guess, ", diff = ", k_guess - k_supply)
    return k_guess - k_supply
end

# solve for eq K
@time k_eq = find_zero(k -> find_eq_k(k, p), (p.kmin, p.kmax), atol = p.tol_k);

# save
result = solve_eq(k_eq, p); # k_supply, λ, v, a_prime, r, w

println("Equilibrium capital = ", k_eq, ", r = ", result[5],", w = ", result[6] )  
plot(p.k_vec, sum(result[2],dims=1)')