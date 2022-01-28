using Distributions: Categorical

p = [0.1,0.7,0.2]

choices = [rand(Categorical(p)) for i in 1:10000]
freq = [count(choices .== i) for i = 1:3]/10000
