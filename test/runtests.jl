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

function generate_fake_img()
    stack = zeros(256, 256, 256)
    stack = create_sphere(stack, (128, 128, 128), 96, 128)
    stack = convert(Array{Float64}, stack)
    return stack
end

function Z_test(Zcoeffs, img, initial_param::pd_density.InitialParam)
    img_ = pd_density.construct_Zernimg(Zcoeffs, img, initial_param)
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Zcoeffs atol = 1e-4
    return result
end

@testset "pd_density.jl" begin

    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 9 # Z(1,-1) -> Z(4,4)

    # test empty image
    img = zeros(32, 32, 32)
    img[16, 16, 16] = 1.0
    initial_param = pd_density.InitialParam(n, NA, lambda, Z_orders)
    Z = zeros(Z_orders)

    # test construct_Zernimg()
    img_ = pd_density.construct_Zernimg(Z, img, initial_param)
    Zk = copy(Z)
    Zk[7] = 3 * lambda
    imgstack = zeros(Complex{Float64}, (size(img)..., 2))
    imgstack[:, :, :, 1] = img_
    imgstack[:, :, :, 2] = pd_density.construct_Zernimg(Zk, img_, initial_param, true)

    Hz, Zval = pd_density.construct_Zernmat(initial_param, size(img_))
    Hk, Dk, Sk, _, ukeep, _, _ = pd_density.loss_prep(Z, imgstack, Hz, Zval, Zk)
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
    Zcol = []
    push!(Zcol, copy(Z))
    push!(Zcol, copy(Zk))
    result = zernike_img_fit(imgstack, initial_param; Zcol, g_abstol = 1e-6)

    #@test Optim.minimizer(result) ≈ Z atol = 1e-4

    # test fake img with psf effect
    result = zernike_img_fit(img_, initial_param; F = img, g_abstol = 1e-6)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4

    # test fake img with Zernike coefficient
    for i = 1:Z_orders
        Zcoeffs = copy(Z)
        Zcoeffs[i] = 3.0
        Z_test(Zcoeffs, img, initial_param)
    end
end
