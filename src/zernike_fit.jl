using FFTW, ZernikePolynomials, DSP
include("pd_initialize.jl")
include("supzern.jl")

export zernike_img_fit


function zernikeloss!(g, Z, img, Hz, Zval)
    imsz = size(img)
    phi = Zcoefs2phi(Z, Zval) #calculate phi with current Zcoefs

    Hk = zeros(Complex{Float64}, imsz) # compute the pupil function
    for i = 1:imsz[3]
        Hk[:, :, i] = Hz[:, :, i] .* exp(im * (phi))
    end
    hk = ifft(Hk)
    sk = hk .* conj(hk)
    Sk = fft(sk) # PSF in coherent illumination

    #compute transform of image
    Dk = fft(img)

    # penalty prep
    S2tot = Sk .* conj(Sk)
    ukeep = abs.(S2tot) .> eps()
    D2tot = Dk .* conj(Dk)
    DdotS = Dk .* conj(Sk)

    num = DdotS .* conj(DdotS)

    if g !== nothing
        # TODO debug gradient function
        coef1 = S2tot .* DdotS
        coef2 = DdotS .* conj(DdotS)
        grad_num = coef1 .* conj(Dk) - coef2 .* conj(Sk)
        Zk = reshape(grad_num[ukeep] ./ S2tot[ukeep] .^ 2, size(grad_num))
        ZconvH = fft(ifft(Zk) .* ifft(conj(Hk)))

        g = 4 * imag(sum(Hk .* ZconvH))
    end

    return -sum(num[ukeep] ./ S2tot[ukeep]) + sum(D2tot)
end


function zernike_img_fit(img, initial_param; kwargs...)
    n, NA, lambda, imsz, Z_orders = initial_param
    H, rho = pd_initial(NA, lambda, imsz)
    Zval = zernike_value(H, Z_orders)
    Hz = zern_initial(img, H, rho, initial_param)

    g = nothing
    f(Z) = zernikeloss!(g, Z, img, Hz, Zval)
    function g!(g,Z)
        g = zeros(1, Z_orders)
        zernikeloss!(g, Z, img, Hz, Zval)
        return g
    end
    params = zeros(1, Z_orders)

    #result = optimize(f, g!, params, BFGS(), Optim.Options(; kwargs...))
    result = optimize(f, params, BFGS(), Optim.Options(; kwargs...))
    Optim.converged(result) || @warn "Optimization failed to converge"
    return result
end
