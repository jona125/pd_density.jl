
function loss_prep(Z, img, Hz, Zval, Z_de)
    imsz = size(img)[1:3]
    Hk = zeros(Complex{Float64}, (imsz..., 2))
    Sk = zeros(Complex{Float64}, (imsz..., 2))
    Hk[:, :, :, 1], Sk[:, :, :, 1] = ZernFT(Z, Hz, Zval, imsz)
    #@show Z
    Z_de .+= Z
    Hk[:, :, :, 2], Sk[:, :, :, 2] = ZernFT(Z_de, Hz, Zval, imsz)

    #compute transform of image
    Dk = zeros(Complex{Float64}, (imsz..., 2))
    for i = 1:2
        Dk[:, :, :, i] = fft(img[:, :, :, i])
    end

    # penalty prep
    S2tot = dropdims(sum(abs2.(Sk), dims = 4), dims = 4)
    ukeep = S2tot .> eps()
    D2tot = dropdims(sum(abs2.(Dk), dims = 4), dims = 4)
    DdotS = dropdims(sum(Dk .* conj(Sk), dims = 4), dims = 4)
    return Hk, Dk, Sk, S2tot, ukeep, D2tot, DdotS
end

function zernikeloss(Z, img, Hz, Zval, Z_de)
    _, _, _, S2tot, ukeep, D2tot, DdotS = loss_prep(Z, img, Hz, Zval, Z_de)
    num = abs2.(DdotS)

    @show -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
    return -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
end

function zernikegrad!(g, Z, img, Hz, Zval, Z_de)
    Hk, Dk, Sk, S2tot, ukeep, _, DdotS = loss_prep(Z, img, Hz, Zval, Z_de)

    coef1 = S2tot .* DdotS
    coef2 = DdotS .* conj(DdotS)
    grad_num = zeros(Complex{Float64}, size(img)[1:3])
    Zk = zeros(Complex{Float64}, size(img))
    ZconvH = zeros(Complex{Float64}, size(img))
    for i = 1:2
        grad_num[:, :, :] = coef1 .* conj(Dk[:, :, :, i]) - coef2 .* conj(Sk[:, :, :, i])
        for id in findall(ukeep)
            idk = CartesianIndex(id, i)
            Zk[idk] = grad_num[id] / (S2tot[id] .^ 2)
        end
        ZconvH[:, :, :, i] = fft(ifft(Zk[:, :, :, i]) .* ifft(conj(Hk[:, :, :, i])))
    end

    grad_mat = 4 * imag(sum(Hk .* ZconvH, dims = 4))
    for id in eachindex(g)
        g[id] = sum(grad_mat .* Zval[:, :, id])
    end
    #@show g
end

function zernike_img_fit(img, initial_param::InitialParam; kwargs...)
    Hz, Zval, _ = construct_Zernmat(initial_param, size(img))

    Zcoeffs = zeros(1, initial_param.Z_orders)
    Zcoeffs[2] = 7 * initial_param.lambda

    #g = zeros(1, Z_orders)
    imgstack = zeros(size(img)..., 2)
    imgstack[:, :, :, 1] = img
    imgstack[:, :, :, 2] = construct_Zernimg(Zcoeffs, img, initial_param, true)
    f(Z) = zernikeloss(Z, imgstack, Hz, Zval, Zcoeffs)
    g!(g, Z) = zernikegrad!(g, Z, imgstack, Hz, Zval, Zcoeffs)

    params = zeros(1, Z_orders)

    #result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
