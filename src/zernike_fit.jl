using FFTW, ZernikePolynomials, DSP
include("pd_initialize.jl")  # in general, I'd recommend putting all your `include`s in the top-level source file (`pd_density.jl`)
include("suppzern.jl")       # if you need to move the `using` too, that's fine

export zernike_img_fit       # I'd also put the `exports` there. It makes it easier to see at a glance what a package exports

function loss_prep!(Z, img, Hz, Zval)   # does this function actually modify any of its arguments? If not, why is it a `!` function?
    imsz = size(img)
    Hk, Sk = ZernFF(Z, Hz, Zval, imsz)

    #compute transform of image
    Dk = fft(img)

    # penalty prep
    S2tot = Sk .* conj(Sk)
    ukeep = abs.(S2tot) .> eps()
    D2tot = Dk .* conj(Dk)
    DdotS = Dk .* conj(Sk)
    return Hk, Dk, Sk, S2tot, ukeep, D2tot, DdotS
end

function zernikeloss!(Z, img, Hz, Zval)   # likewise, does it modify outputs?
    _, _, _, S2tot, ukeep, D2tot, DdotS = loss_prep!(Z, img, Hz, Zval)
    num = DdotS .* conj(DdotS)

    return -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
end

function zernikegrad!(g, Z, img, Hz, Zval)
    Hk, Dk, Sk, S2tot, ukeep, _, DdotS = loss_prep!(Z, img, Hz, Zval)

    coef1 = S2tot .* DdotS
    coef2 = DdotS .* conj(DdotS)
    grad_num = coef1 .* conj(Dk) - coef2 .* conj(Sk)
    Zk = zeros(Complex{Float64}, size(grad_num))
    for id in findall(ukeep)
        Zk[id] = grad_num[id] ./ (S2tot[id] .^ 2)
    end
    ZconvH = fft(ifft(Zk) .* ifft(conj(Hk)))

    grad_mat = 4 * imag(Hk .* ZconvH)
    for id in eachindex(g)
        g[id] = sum(grad_mat .* Zval[:, :, id])
    end
    return g    # I generally like to return the modified argument, because it allows you to pass it on to downstream code, e.g., `norm(zernikegrad!(...))`
    #@show g
end

function zernike_img_fit(img, initial_param; kwargs...)
    # `initial_param` encodes meaning-by-position, which I don't recommend---it makes it hard to modify and hard to understand
    # Good alternatives:
    # 1. Use a NamedTuple: `initial_param = (n = n, NA = NA, lambda = lambda, imsz = size(img), Z_orders = Z_orders)`
    # 2. Use a struct: `struct InitialParam; n; NA; lambda; imsz; Z_orders; end` except give each parameter types
    # With either one, Julia lets you unpack them very efficiently:
    #    (; n, NA, lambda, imsz, Z_orders) = initial_param
    # will extract the fields of `initial_param` into the variables `n`, `NA`, `lambda`, `imsz`, and `Z_orders`
    n, NA, lambda, imsz, Z_orders = initial_param
    H, rho, theta = pd_initial(NA, lambda, imsz)
    Zval = zernike_value(H, Z_orders, rho, theta)
    Hz = zern_initial(img, H, rho, initial_param)

    #g = zeros(1, Z_orders)
    f(Z) = zernikeloss!(Z, img, Hz, Zval)
    g!(g, Z) = zernikegrad!(g, Z, img, Hz, Zval)

    params = zeros(1, Z_orders)

#result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
