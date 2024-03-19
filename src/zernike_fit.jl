# Fitting function zernike_img_fit()
# Phase diversity Loss function zernikeloss()
# Phase diversity Gradient function zernikegrad!()
# PSF based Loss function psfloss()
# PSF based Gradient function psfgrad!()
# Matrix preparation function loss_prep()
#

function loss_prep(Z, img, Hz, Zval, Z_de)
    imsz = size(img)[1:3]
    K = length(size(img)) == 4 ? size(img)[4] : 1
    Hk = zeros(Complex{Float64}, (imsz..., K))
    Sk = zeros(Complex{Float64}, (imsz..., K))
    for i = 1:K
        Hk[:, :, :, i], Sk[:, :, :, i] = ZernFT(Z .+ Z_de[i], Hz, Zval, imsz)
    end

    #compute transform of image
    Dk = fft(img; dims=1:3)

    # penalty prep
    S2tot = dropdims(sum(abs2.(Sk), dims = 4), dims = 4)
    ukeep = S2tot .> eps()
    D2tot = dropdims(sum(abs2.(Dk), dims = 4), dims = 4)
    DdotS = dropdims(sum(Dk .* conj(Sk), dims = 4), dims = 4)
    return Hk, Dk, Sk, S2tot, ukeep, D2tot, DdotS
end

# Phase diversity loss and gradient function
function zernikeloss(Z, img, Hz, Zval, Z_de)
    _, _, _, S2tot, ukeep, D2tot, DdotS = loss_prep(Z, img, Hz, Zval, Z_de)
    num = abs2.(DdotS)

    return -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
end

function zernikegrad!(g, Z, img, Hz, Zval, Z_de)
    Hk, Dk, Sk, S2tot, ukeep, _, DdotS = loss_prep(Z, img, Hz, Zval, Z_de)

    K = size(img)[4]
    coef1 = S2tot .* DdotS
    coef2 = DdotS .* conj(DdotS)
    grad_num = zeros(Complex{Float64}, size(img)[1:3])
    Zk = zeros(Complex{Float64}, size(img))
    ZconvH = zeros(Complex{Float64}, size(img))
    for i = 1:K
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

# Known F spread loss and gradient function
function psfloss(Z, img, Hz, Zval)
    _, Dk, Sk, _, _, _, _ = loss_prep(Z, img, Hz, Zval, [Z])

    F = zeros(size(img))
    F[CartesianIndex(Int.(floor.(size(img) ./ 2)))] = 1.0
    P = Dk[:, :, :, 1] .- fft(F) .* Sk[:, :, :, 1]

    @show sum(abs2.(P))
    return sum(abs2.(P))
end

function psfgrad!(g, Z, img, Hz, Zval)
    Hk, Dk, Sk, _, _, _, _ = loss_prep(Z, img, Hz, Zval, [Z])

    F = zeros(size(img))
    F[CartesianIndex(Int.(floor.(size(img) ./ 2)))] = 1.0
    F = fft(F)
    F2tot = abs2.(F)
    H = dropdims(Hk, dims = 4)
    S = dropdims(Sk, dims = 4)
    D = dropdims(Dk, dims = 4)

    Z1convH = fft(ifft(D .* conj(F)) .* ifft(conj(H)))
    Z2convH = fft(ifft(F2tot .* conj(S)) .* ifft(conj(H)))

    grad = 2 .* (imag(H .* Z2convH .- H .* Z1convH .* 2))
    for id in eachindex(g)
        g[id] = sum(grad .* Zval[:, :, id])
    end
    return g
end

function zernike_img_fit(img, initial_param::InitialParam; Zcol = [], kwargs...)
    Hz, Zval, _ = construct_Zernmat(initial_param, size(img)[1:3])
    f(Z) = !isempty(Zcol) ? zernikeloss(Z, img, Hz, Zval, Zcol) : psfloss(Z, img, Hz, Zval)
    g!(g, Z) =
        !isempty(Zcol) ? zernikegrad!(g, Z, img, Hz, Zval, Zcol) :
        psfgrad!(g, Z, img, Hz, Zval)

    params = zeros(1, Z_orders)

    grad = 2 .* imag(H .* Z2convH .- H .* Z1convH)
    for id in eachindex(g)
        g[id] = sum(grad .* Zval[:, :, id])
    end
    return g
end

function zernike_img_fit(
    img,
    initial_param::InitialParam;
    Zcol = [],
    F = zeros(size(img)),
    kwargs...,
)
    Hz, Zval = construct_Zernmat(initial_param, size(img)[1:3])
    f(Z) =
        !isempty(Zcol) ? zernikeloss(Z, img, Hz, Zval, Zcol) : psfloss(Z, img, Hz, Zval, F)
    g!(g, Z) =
        !isempty(Zcol) ? zernikegrad!(g, Z, img, Hz, Zval, Zcol) :
        psfgrad!(g, Z, img, Hz, Zval, F)

    params = zeros(initial_param.Z_orders)
    #result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
