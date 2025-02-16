using DrWatson, Test
@quickactivate "redox_steady_state_aquifers"

# Here you include files using `srcdir`
# include(srcdir("file.jl"))

# Run test suite
println("Starting tests")
ti = time()

@testset "redox_steady_state_aquifers tests" begin
    include("test_conservative_transp.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti/60, digits = 3), " minutes")
