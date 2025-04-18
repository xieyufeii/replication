"""
210B HW3 occupation choice question
Author: Yufei Xie, yux085@ucsd.edu
"""

using Plots, Parameters, SparseArrays, LinearAlgebra, Statistics, Roots
using Distributions, Expectations, Plots, BenchmarkTools, DelimitedFiles

@with_kw mutable struct model
    neps::Int64 = 11;
    epsmin::Float64 = 0.01;
    epsmax::Float64 = 1.0;
    epsg = collect(range(epsmin, epsmax, neps));
    # epsprob = ones(neps)./neps;
    epsprob = expectation(BetaBinomial(neps-1,2,6)).weights;

    na::Int64 = 100;
    amin::Float64 = 0.0;
    amax::Float64 = 10.0;
    ag = collect(range(amin, amax, na));
    gam::Float64 = 0.5;

    ns::Int64 = 2;
    sg = [0.5 1.5]';
    sprob = [0.6 0.4];
    P_s = [sprob;sprob];

    sig::Float64 = 2;  # CRRA Parameter
    beta::Float64 = 0.96;
    alph::Float64 = 0.3;
    delta::Float64 = 0.0125;
    nu::Float64 = 0.85;

    max_iter_v::Int64 = 1000;
    max_iter_λ::Int64 = 1000;
    max_iter_p::Int64 = 20;
    tol_v::Float64 = 1e-5;
    tol_λ::Float64 = 1e-10;
    tol_p::Float64 = 1e-4;
    upda::Float64 = 0.7;
    Delta::Float64 = 50;

end
p = model();

# utility
function u(c::Float64, ρ::Float64)
    c = max.(c,1e-10);
    if ρ != 1
        u = (c^(1-ρ))/(1-ρ)
    else
        u = log(c)
    end
    return u
end

# state: (a,s,eps)=(asset, idiosyncratic shock, invariant productivity)

# entrepreneurs' problem
function compute_πkn(param,w,r)
    @unpack_model param
    k_z_tmp1 = ((r+delta)/(alph*nu*(((1-alph)*(r+delta)/(w*alph))^((1-alph)*nu))))^(1/(nu-1)).*epsg.^(1/(1-nu));
    k_star = repeat(reshape(k_z_tmp1,1,1,neps),na,ns);
    k_z_tmp2 = (1+gam).*ag;
    k_bar = repeat(k_z_tmp2,1,ns,neps);
    k = min.(k_star,k_bar);
    n = max.(((r+delta)*(1-alph)/(w*alph)).*k,repeat(reshape(sg,1,ns,1),na,1,neps));
    k_star2 = ((w*alph)/(r+delta*(1-alph))).*n;
    k = min.(k_star2,k_bar); # not sure if this is correct
    Pi = (k.^(alph*nu)).*(n.^((1-alph)*nu)).*reshape(epsg,1,1,neps)-(r+delta).*k+(1+r).*repeat(ag,1,ns,neps)-w.*(n.-reshape(sg,1,ns,1));
    kd = k;
    ld = n.-reshape(sg,1,ns,1);
    return Pi, kd, ld
end

# state: (a,s,eps)=(asset, idiosyncratic shock, invariant productivity)

function solve_eq(param)

    @unpack_model param

    # initial guess
    w = repeat(collect(range(0.4,1,na)),1,ns,neps); # worker
    v = repeat(collect(range(0.1,2,na)),1,ns,neps); # entrepreneur
    λ = ones(na,ns,neps)./sum(ones(na,ns,neps));
    a_w_prime = zeros(na,ns,neps);
    a_v_prime = zeros(na,ns,neps);
    o_prime = zeros(na,ns,neps).+[zeros(Int(floor(na/2)),1);ones(na-Int(floor(na/2)),1)];  # 0-worker, 1-entrepreneur
    w_tmp = zeros(na,ns,neps,na); # d4: a' choice
    v_tmp = zeros(na,ns,neps,na);
    c_w_tmp = zeros(na,ns,neps,na);
    c_v_tmp = zeros(na,ns,neps,na);
    u_w_tmp = zeros(na,ns,neps,na);
    u_v_tmp = zeros(na,ns,neps,na);

    wage = 0.85;
    r = 0.02;
    w_next = 0;
    r_next = 0;
    ip = 0;

    while (norm(wage-w_next) > tol_p || norm(r-r_next) > tol_p ) && ip < max_iter_p

        if ip > 0
            wage = w_next;
            r = r_next;
        end

        println("prices:    $ip iterations with wage = $wage, interest rate = $r")

        # with new w,r
        Pi,kd,ld = compute_πkn(p,wage,r);

        # consumption
        c_w_tmp= wage.*reshape(sg,1,ns,1,1) .+ (1+r).*reshape(ag,na,1,1,1) .- reshape(ag,1,1,1,na) .+ reshape(zeros(neps,1),1,1,neps,1);
        c_v_tmp= Pi .- reshape(ag,1,1,1,na);
        u_w_tmp = u.(c_w_tmp, sig);
        u_v_tmp = u.(c_v_tmp, sig);

        # value function iterations
        function Tv(w,v)
            Ew = reshape(sum(w.*reshape(sprob,1,ns,1), dims=2),na,neps); # (na,neps)
            Ev = reshape(sum(v.*reshape(sprob,1,ns,1), dims=2),na,neps); # (na,neps)
            o_prime = repeat(reshape((Ev .> Ew) .*1,na,1,neps),1,ns,1); # Int64
            Eu = (1 .-o_prime).*repeat(reshape(Ew',1,1,neps,na),na,ns,1,1) +
                o_prime.*repeat(reshape(Ev',1,1,neps,na),na,ns,1,1); # check, (na,ns,neps,na)
            w_tmp = u_w_tmp .+ beta*Eu;
            v_tmp = u_v_tmp .+ beta*Eu;

            w_prime, a_w_prime = findmax(w_tmp, dims=4);
            w_prime = dropdims(w_prime, dims=4);
            a_w_prime = dropdims(a_w_prime, dims=4) .|> x -> x[4]

            v_prime, a_v_prime = findmax(v_tmp, dims=4);
            v_prime = dropdims(v_prime, dims=4);
            a_v_prime = dropdims(a_v_prime, dims=4) .|> x -> x[4]
            return w_prime,v_prime,a_w_prime,a_v_prime,o_prime
        end

        w_next = similar(w)
        v_next = similar(v)
        i = 0;
        Error = 1;
        while i < max_iter_v && Error > tol_v
            w_next,v_next,a_w_next,a_v_next,o_prime = Tv(w,v)
            Error = max.(norm(w_next.-w),norm(v_next.-v))
            # if i % 10 == 0
            #     println("w,v:    $i iterations with error $Error")
            # end
            w .= (1-upda)*w + upda*w_next
            v .= (1-upda)*v + upda*v_next
            a_w_prime = a_w_next
            a_v_prime = a_v_next
            i = i+1;
        end
        # heatmap(o_prime[:,1,:]')

        # stationary distribution (na,ns,neps)
        function Tλ(λ)
            λ_w = λ.*(1 .-o_prime);
            λ_v = λ.*o_prime;
            S_w = reshape(sparse(1:na*ns*neps,vec(a_w_prime),vec(λ_w),na*ns*neps,na),na,ns,neps,na);
            S_v = reshape(sparse(1:na*ns*neps,vec(a_v_prime),vec(λ_v),na*ns*neps,na),na,ns,neps,na);
            λ_w_tmp = reshape(permutedims(sum(S_w,dims=1),(4,2,3,1)),na,ns,neps);
            λ_v_tmp = reshape(permutedims(sum(S_v,dims=1),(4,2,3,1)),na,ns,neps);
            λ_prime = reshape(permutedims((λ_w_tmp+λ_v_tmp),(1,3,2)),na*neps,ns)*P_s;
            λ_prime = permutedims(reshape(λ_prime,na,neps,ns),(1,3,2));
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

        # agg capital, labor
        k_supply = sum(repeat(ag,1,ns,neps).*λ);
        k_demand = sum(kd.*o_prime.*λ);
        r_next = r + (k_demand-k_supply)/Delta;
        n_supply = sum((1 .-o_prime).*λ.*reshape(sg,1,ns,1));
        n_demand = sum(ld.*o_prime.*λ);
        w_next = wage + (n_demand-n_supply)/Delta;

        ip = ip + 1;
    end

    println("Equilibrium wage = $wage, interest rate = $r")

    return wage,r,λ,o_prime,w,v,a_w_prime,a_v_prime
end

result = solve_eq(p) # wage,r,λ,o_prime,w,v,a_w_prime,a_v_prime

# value functions
v = result[6];
w = result[5];
plot(p.ag[2: p.na],w[2: p.na,1,6],lw=2,label="worker value function",xlabel="asset",title="Value functions (ϵ grid=6)",size=(500,390))
plot!(p.ag[2: p.na],v[2: p.na,1,6],lw=2,label="entrepreneur value function")


# occupation choice
o_prime = result[4];
epsidx = (-1).*(o_prime.*repeat(reshape(collect(1:p.neps),1,1,p.neps),p.na,p.ns,1)+(1 .-o_prime).*9999)
_,z_bar = findmax(epsidx, dims=3);
z_bar = dropdims(z_bar, dims=3) .|> x -> x[3];
plot(p.ag[2: p.na],z_bar[2: p.na,1],lw=2,label="ϵ grid",xlabel="asset",title="ϵ threshold",ylims=(0,12),size=(500,390))


# wealth distribution
λ = result[3];
plot(p.ag,reshape(sum(λ,dims=(2,3)),p.na,1),label="",title="Unconditional wealth distribution",size=(550,400),lw=1.5)
plot(p.ag,λ[:,1,1]./sum(λ[:,1,1]),title="conditional wealth distribution (s given)", label="ϵ=low",size=(530,400),lw=1.5)
plot!(p.ag,λ[:,1,8]./sum(λ[:,1,8]),label="ϵ=high",lw=1.5)
plot!(p.ag,λ[:,1,9]./sum(λ[:,1,9]),label="ϵ=very high",lw=1.5)