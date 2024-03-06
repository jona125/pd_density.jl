# Fitting function zernike_img_fit()
# Loss function zernikeloss()
# Gradient function zernikegrad!()
# Matrix preparation function loss_prep()
#

function loss_prep(Z, img, Hz, Zval)
    imsz = size(img)
    Hk, Sk = ZernFT(Z, Hz, Zval, imsz)

    #compute transform of image
    Dk = fft(img)

    # penalty prep
    S2tot = Sk .* conj(Sk)
    ukeep = abs.(S2tot) .> eps()
    D2tot = Dk .* conj(Dk)
    DdotS = Dk .* conj(Sk)
    return Hk, Dk, Sk, S2tot, ukeep, D2tot, DdotS
end

function zernikeloss(Z, img, Hz, Zval)
    _, _, _, S2tot, ukeep, D2tot, DdotS = loss_prep(Z, img, Hz, Zval)
    num = DdotS .* conj(DdotS)

    return -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
end

function zernikegrad!(g, Z, img, Hz, Zval)
    Hk, Dk, Sk, S2tot, ukeep, _, DdotS = loss_prep(Z, img, Hz, Zval)

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
    return g
    #@show g
end

function zernike_img_fit(img, initial_param::InitialParam; kwargs...)
    Hz, Zval = construct_Zernmat(initial_param, img)

    #g = zeros(1, Z_orders)
    f(Z) = zernikeloss(Z, img, Hz, Zval)
    g!(g, Z) = zernikegrad!(g, Z, img, Hz, Zval)

    params = zeros(1, initial_param.Z_orders)

    result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    #result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
