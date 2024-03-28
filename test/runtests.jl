using pd_density
using Test
using Optim, FiniteDifferences, FFTW, Statistics

function Z_test(Zcoeffs, img, initial_param::pd_density.InitialParam, noise = 0.0)
    img_ = pd_density.construct_Zernimg(Zcoeffs, img, initial_param)
    img_ .+= noise * mean(img_) .* rand(size(img)...)
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Zcoeffs atol = 1e-4
    return result
end

@testset "pd_density.jl" begin
    # initial parameters
    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 9 # Z(1,-1) -> Z(3,3)
    pixel_spacing = [1.0, 1.0, 1.0]

    # test empty image
    img = zeros(32, 32, 32)
    img[16, 16, 16] = 1.0
    initial_param = pd_density.InitialParam(n, NA, lambda, Z_orders, pixel_spacing)
    Z = zeros(Z_orders)
    Zcol = []
    push!(Zcol, copy(Z))
    push!(Zcol, copy(Z))

    # test construct_Zernimg()
    img_ = pd_density.construct_Zernimg(Zcol[1], img, initial_param)
    Zcol[2][7] += lambda
    imgstack = zeros(Complex{Float64}, (size(img)..., 2))
    imgstack[:, :, :, 1] = img_
    imgstack[:, :, :, 2] = pd_density.construct_Zernimg(Zcol[2], img, initial_param)

    Hz, Zval = pd_density.construct_Zernmat(initial_param, size(img_))
    _, Dk, Sk, _, ukeep, _, _ = pd_density.loss_prep(Z, imgstack, Hz, Zval, Zcol)
    F = zeros(Complex{Float64}, (size(img)..., 2))
    for i = 1:2
        for id in findall(ukeep)
            idx = CartesianIndex(id, i)
            F[idx] = (Dk[idx] .* conj(Sk[idx])) / abs2.(Sk[idx])
        end
    end
    @test img ≈ abs.(ifft(F[:, :, :, 1])) rtol = 1
    @test img_ ≈ abs.(ifft(F[:, :, :, 1] .* Sk[:, :, :, 1])) atol = 0.1
    @test imgstack[:, :, :, 2] ≈ abs.(ifft(F[:, :, :, 2] .* Sk[:, :, :, 2])) atol = 0.1

    # test fake img with K phase diversity image
    result = zernike_img_fit(imgstack, initial_param; Zcol, g_abstol = 1e-14)
    @test Optim.minimizer(result) ≈ Z atol = 1e-6

    # test denstiy gradient function
    f_d(X) = pd_density.zernikeloss(X, imgstack, Hz, Zval, Zcol)
    g_d!(g, X) = pd_density.zernikegrad!(g, X, imgstack, Hz, Zval, Zcol)  
    g = zeros(Z_orders)
    @test g_d!(g, Z) ≈ grad(central_fdm(5, 1), f_d, Z)[1] atol = 1e-8

    # test psf gradient function
    f_p(X) = pd_density.psfloss(X, imgstack[:, :, :, 1], Hz, Zval, img)
    g_p!(g, X) = pd_density.psfgrad!(g, X, imgstack[:, :, :, 1], Hz, Zval, img)
    g = zeros(Z_orders)
    @test g_p!(g, Z) ≈ grad(central_fdm(2, 1), f_p, Z)[1] atol = 1e-14

    # test fake img with psf effect
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-6)
    @test Optim.minimizer(result) ≈ Z atol = 1e-4

    # test fake img with Zernike coefficient
    for i = 1:Z_orders
        Zcoeffs = copy(Z)
        Zcoeffs[i] = 1.0
        Z_test(Zcoeffs, img, initial_param)
        noise = 0.02
        Z_test(Zcoeffs, img, initial_param, noise)
    end

    # test random Zernike coefficient
    Zcoeffs = rand(Z_orders) ./ 2.0
    Z_test(Zcoeffs, img, initial_param)
end
