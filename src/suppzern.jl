# Fourier transfer plane generation ZernFT()
# Zernike collection generation zernike_value()
# Zernike polynomial Zern_gen()
# phi generation from Zernike value Zcoefs2phi()
# construct matrices for fitting construct_Zernmat()
#


function ZernFT(Z, Hz, Zval, imsz)
    phi = Zcoefs2phi(Z, Zval)

    Hk = zeros(Complex{Float64}, imsz) # compute the pupil function
    hk = zeros(Complex{Float64}, imsz)
    for i = 1:imsz[3]
        Hk[:, :, i] = Hz[:, :, i] .* cis.(phi)
        hk[:, :, i] = ifft(Hk[:, :, i])
    end
    sk = abs2.(hk)
    Sk = zeros(Complex{Float64}, imsz)
    for i = 1:imsz[3]
        Sk[:, :, i] = fft(sk[:, :, i])
    end

    return Hk, Sk
end


function zernike_value(H, n_coefs, rho, theta)
    rholim = copy(rho)
    rholim[rholim.>1] .= 0
    Zval = zeros((size(H)..., n_coefs))
    for ix = 1:n_coefs
        Zval[:, :, ix] = Zern_gen(ix, size(H), rholim, theta) .* H
    end
    return Zval
end

function Zern_gen(p, sz, rho, theta)
    n = Int(ceil((-3 + sqrt(9 + 8 * p)) / 2))
    m = Int(2 * p - n * (n + 2))

    Z = zernike(NM(n, m); coord = :polar)

    return Z.(rho, theta)
end

function Zcoefs2phi(Zcoefs, Zval)
    #@show Zcoefs
    phi = zeros(size(Zval)[1:end-1])
    for i = 1:length(Zcoefs)
        phi .= phi .+ Zcoefs[i] .* Zval[:, :, i]
    end
    return phi
end


function construct_Zernimg(Zcoefs, img, initial_param::InitialParam)
    Hz, Zval = construct_Zernmat(initial_param, size(img))
    _, S = ZernFT(Zcoefs, Hz, Zval, size(img))

    imgfft = fft(img)
    D = imgfft .* S

    return (ifft(D))
end

function construct_Zernmat(initial_param::InitialParam, imsz)
    (; n, NA, lambda, Z_orders, pixel_spacing) = initial_param
    H, rho, theta = pd_initial(NA, lambda, imsz)
    Zval = zernike_value(H, Z_orders, rho, theta)
    Hz = zern_initial(H, rho, initial_param, imsz)

    return Hz, Zval
end
