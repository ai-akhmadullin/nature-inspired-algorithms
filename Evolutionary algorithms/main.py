import numpy as np
import random
import matplotlib.pyplot as plt
# import copy

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--file", default="debugging_data_10.txt", type=str, help="Path to the file with data")

def evaluate_fitness(individual, data, max_volume):
    value = np.sum(individual * data[:, 0])
    volume = np.sum(individual * data[:, 1])
    if volume > max_volume:
        return 0,
    else:
        return value,

# Possible own implementation of mutation (with this mutation, we can avoid zeroing the fitness values 
# in case of exceeding the permitted volume, since it will not happen)

# def mutation(individual, indpb, data, max_volume):
#     old = copy.deepcopy(individual)
#     for i in range(0, len(individual)):
#         if random.random() < indpb:
#             individual[i] = not individual[i]
#     if np.sum(individual * data[:, 1]) > max_volume:
#         return old,
#     return individual,

# Possible own implementation of crossover (with this crossover, we can avoid zeroing the fitness values 
# in case of exceeding the permitted volume, since it will not happen)

# def crossover(individual1, individual2, data, max_volume):
#     old1 = copy.deepcopy(individual1)
#     old2 = copy.deepcopy(individual2)
#     child1 = individual1
#     child2 = individual2
#     idx = random.randint(0, len(individual1) - 1)
#     child1[idx], child2[idx] = child2[idx], child1[idx]
#     if np.sum(child1 * data[:, 1]) > max_volume:
#         child1 = old1
#     if np.sum(child2 * data[:, 1]) > max_volume:
#         child2 = old2
#     return child1, child2

# This approach did not show a good result.

def main(args):
    with open(args.file, 'r') as f:
        line = f.readline().split()
        N, V = int(line[0]), int(line[1])
        data = f.read().splitlines()
        data = np.array([line.split() for line in data], dtype=int)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("zeroOne", lambda : 0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.zeroOne, n=N)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_fitness, data=data, max_volume=V)
    toolbox.register("mate", tools.cxUniform, indpb=0.1)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=N*15)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, logs = algorithms.eaSimple(pop, toolbox, 
                                        cxpb=0.8, mutpb=0.2, ngen=100,
                                        stats=stats, halloffame=hof, 
                                        verbose=True)

    best = hof[0]
    value = np.sum(best * data[:, 0])
    volume = np.sum(best * data[:, 1])
    print(f"Best solution: {best}\nWith value: {value}\nTaken volume: {volume}")

    plt.plot(logs.select("max"), label = "Maximum fitness")
    plt.plot(logs.select("avg"), label = "Average fitness")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)