import statistics
import neat
import os
import matplotlib.pyplot as plt
import random
import timeit

local_dir = os.path.dirname(__file__)
config_file = os.path.join(local_dir, "config-feedforward")

results = []

def_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_file,
)

# Define your actual fitness function here
def calculate_fitness(genome, config):
    # Replace this with your actual fitness calculation
    # Ensure it always returns a valid float value
    return random.random()

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = calculate_fitness(genome, config)

def run_evolution(config, fitness_threshold, max_generations=1000):
    population = neat.Population(config)
    
    best_fitness = 0
    stagnation_counter = 0
    stagnation_limit = 15  # Number of generations without improvement before stopping

    for generation in range(max_generations):
        population.run(eval_genomes, 1)
        
        best_genome = max(population.population.values(), key=lambda x: x.fitness if x.fitness is not None else float('-inf'))
        
        if best_genome.fitness is not None and best_genome.fitness > best_fitness:
            best_fitness = best_genome.fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        if best_fitness >= fitness_threshold or stagnation_counter >= stagnation_limit:
            return best_fitness, generation + 1  # +1 because generations are 0-indexed

    return best_fitness, max_generations

i = 100
y = []
generations = []

for i in range(0, i):
    print(f"Run {i+1}/100")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file,
    )
    config.pop_size = (10*i)+10
    y.append(config.pop_size)
    
    # Measure time to reach peak fitness
    time_taken = timeit.timeit(lambda: run_evolution(config, fitness_threshold=0.95), number=1)
    results.append(time_taken)
    
    # You might want to record the number of generations as well
    _, num_generations = run_evolution(config, fitness_threshold=0.95)
    generations.append(num_generations)

print("Time results:", results)
print("Average time:", statistics.mean(results))
print("Time standard deviation:", statistics.stdev(results))
print("Average generations:", statistics.mean(generations))
print("Generations standard deviation:", statistics.stdev(generations))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y, results)
plt.ylabel("Time to reach peak fitness")
plt.xlabel("Population size")
plt.title("Time vs Population Size")
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(y, generations)
plt.ylabel("Generations to reach peak fitness")
plt.xlabel("Population size")
plt.title("Generations vs Population Size")
plt.grid()

plt.tight_layout()
plt.savefig('neat_performance_analysis.png')
plt.savefig('neat_performance_analysis.pdf')
plt.show()