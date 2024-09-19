import statistics
import neat
import os
import matplotlib.pyplot as plt
import math
import random
import sys
import timeit

local_dir = os.path.dirname(__file__)
config_file = os.path.join(local_dir, "config-feedforward")

results = []

# Load configuration.
def_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_file,
)

from pprint import pprint
pprint(vars(def_config))

# Define the eval_genomes function
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = random.random()  # Replace with your actual fitness evaluation

i = 100
y = []

for i in range(0, i):
    print(i)
    config = def_config
    def_config.pop_size = (10*i)+10
    y.append(def_config.pop_size)
    p = neat.Population(config)

    # Modify the timeit call to use a lambda function that includes both p.run and eval_genomes
    results.append(timeit.timeit(lambda: p.run(eval_genomes, 300), number=1))

print(results)
print(statistics.mean(results))
print(statistics.stdev(results))

plt.scatter(y, results)
plt.ylabel("Time to reach max fitness")
plt.xlabel("Population size")
plt.grid()
plt.savefig('graph.png')
plt.savefig('Time to reach max fitness vs Population Size.pdf')
plt.show()