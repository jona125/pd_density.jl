using FFTW, ZernikePolynomials, DSP
include("pd_initialize.jl")
include("suppzern.jl")

export zernike_img_fit

function loss_prep!(Z, img, Hz, Zval)
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

function zernikeloss!(Z, img, Hz, Zval)
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
    for id = 1:length(g)
        g[id] = sum(grad_mat .* Zval[:, :, id])
    end
    #@show g
end

function zernike_img_fit(img, initial_param; kwargs...)
    n, NA, lambda, imsz, Z_orders = initial_param
    H, rho, theta = pd_initial(NA, lambda, imsz)
    Zval = zernike_value(H, Z_orders, rho, theta)
    Hz = zern_initial(img, H, rho, initial_param)

    #g = zeros(1, Z_orders)
    f(Z) = zernikeloss!(Z, img, Hz, Zval)
    g!(g, Z) = zernikegrad!(g, Z, img, Hz, Zval)

    params = zeros(1, Z_orders)

    result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    #result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
