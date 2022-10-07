


from fractal_zero.search.fmc import FMC
from fractal_zero.vectorized_environment import SerialVectorizedEnvironment



vec_env = SerialVectorizedEnvironment("CartPole-v0", n=8)
fmc = FMC(vec_env)

fmc.simulate(4)
fmc.tree.render()

fmc.simulate(4)
fmc.tree.render()
