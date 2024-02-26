using pd_density
using Test
using Optim

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
    stack = zeros(256, 256, 64)
    stack = create_sphere(stack, (128, 128, 32), 30, 150)
    stack = convert(Array{Float64}, stack)
    return stack
end


@testset "pd_density.jl" begin

    n = 1.33
    lambda = 0.53
    NA = 0.5
    Z_orders = 7 # Z(2.-2) -> Z(4,4)



    img = generate_fake_img()
    initial_param = n, lambda, NA, size(img), Z_orders
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @show Optim.minimizer(result)


    img = zeros(128, 128, 128)
    img[64, 64, 64] = 1
    initial_param = n, lambda, NA, size(img), Z_orders
    result = zernike_img_fit(img, initial_param; g_abstol = 1e-14)

    @test Optim.minimizer(result) == zeros(1, Z_orders)

end
