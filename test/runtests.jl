using pd_density
using Test
using Optim, ZernikePolynomials, FFTW

function create_sphere(stack, center, radius, gray_level)
    for coord in CartesianIndices(stack)
        if (
            (coord[1] - center[1])^2 + (coord[2] - center[2])^2 + (coord[3] - center[3])^2
        ) <= radius^2
            stack[coord] += gray_level / 256
        end
    end
    return stack
end

function generate_fake_img()
    stack = zeros(256, 256, 256)
    stack = create_sphere(stack, (128, 128, 128), 96, 128)
    stack = convert(Array{Float64}, stack)
    return stack
end


@testset "pd_density.jl" begin

    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 11 # Z(2.-2) -> Z(4,4)

    # test empty image
    img = zeros(32, 32, 32)
    img[15:17, 15:17, 15:17] .= 0.5
    initial_param = pd_density.InitialParam(n, NA, lambda, Z_orders)

    #img = generate_fake_img()
    Z = zeros(1, Z_orders)

    # test construct_Zernimg()
    img_ = pd_density.construct_Zernimg(Z, img, initial_param)
    Zk = copy(Z)
    Zk[2] = 3 * lambda
    imgstack = zeros(size(img)..., 2)
    imgstack[:, :, :, 1] = img_
    imgstack[:, :, :, 2] = pd_density.construct_Zernimg(Zk, img_, initial_param, true)

    Hz, Zval, _ = pd_density.construct_Zernmat(initial_param, size(img_))
    Hk, Dk, Sk, _, ukeep, _, _ = pd_density.loss_prep(Z, imgstack, Hz, Zval, Zk)
    F = zeros(Complex{Float64}, (size(img)..., 2))
    for i = 1:2
        for id in findall(ukeep)
            idx = CartesianIndex(id, i)
            F[idx] = (Dk[idx] .* conj(Sk[idx])) / abs2.(Sk[idx])
        end
    end
    #@test img ≈ abs.(ifft(F)) atol = 1
    @test img_ ≈ abs.(ifft(F[:, :, :, 1] .* Sk[:, :, :, 1])) atol = 0.01
    @test imgstack[:, :, :, 2] ≈ abs.(ifft(F[:, :, :, 2] .* Sk[:, :, :, 2])) atol = 0.01


    # test fake img with psf effect
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ zeros(1, Z_orders) atol = 1e-4


    # test fake img with Zernike coefficient
    Z[5] = 3.0
    img_ = pd_density.construct_Zernimg(Z, img, initial_param)
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4


    # test fake img with random Zernike coefficient
    Z = rand(1, Z_orders) * 3
    img_ = pd_density.construct_Zernimg(Z, img, initial_param)
    result = zernike_img_fit(img_, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) ≈ Z atol = 1e-4

end
