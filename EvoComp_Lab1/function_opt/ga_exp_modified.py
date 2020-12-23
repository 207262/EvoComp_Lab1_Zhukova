from deap import tools, base
from multiprocessing import Pool
from deap.algorithms import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from deap import benchmarks
import pandas as pd
import random

creator.create("BaseFitness", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)

def mutation(individual):
    for random_index in rnd.randint(0, len(individual), 2):
        individual[random_index] += rnd.normal(0.0, 2.0)
        individual[random_index] = np.clip(individual[random_index], -5, 5)
    return individual,

class SimpleGAExperiment:
    def factory(self):
        return rnd.random(self.dimension) * 10 - 5

    def __init__(self, function, dimension, pop_size, iterations, mutation_prob, crossover_prob):
        self.pop_size = pop_size
        self.iterations = iterations
        self.mut_prob = mutation_prob
        self.cross_prob = crossover_prob

        self.function = function
        self.dimension = dimension

        # self.pool = Pool(5)
        self.engine = base.Toolbox()
        # self.engine.register("map", self.pool.map)
        self.engine.register("map", map)
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        self.engine.register("mate", tools.cxOnePoint)
        #self.engine.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
        self.engine.register("mutate", mutation)
        self.engine.register("select", tools.selTournament, tournsize=4)
        # self.engine.register("select", tools.selRoulette)
        self.engine.register("evaluate", self.function)

    def run(self):
        pop = self.engine.population()
        hof = tools.HallOfFame(3, np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size*0.8), cxpb=self.cross_prob, mutpb=self.mut_prob,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log, {'best': hof[0], 'best_fit': hof[0].fitness.values[0]}

from function_opt.functions import rastrigin
if __name__ == "__main__":

    def function(x):
        res = rastrigin(x)
        return res,

    dimension = 100
    pop_size = 100
    iterations = 1000
    mutation_prob = 0.6
    crossover_prob = 0.3
    scenario = SimpleGAExperiment(function, dimension, pop_size, iterations, mutation_prob, crossover_prob)
    log, final_values = scenario.run()
    from draw_log import draw_log
    draw_log(log)
