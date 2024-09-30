from run_game import *
from random import choice, randint

def cal_pop_fitness(pop, filename):

    fitness = []
    for i in range(pop.shape[0]):
        fit = run_game_with_ML(display, clock, pop[i])
        
        fitness.append(fit)
    return np.array(fitness)

def select_mating_pool(pop, fitness, num_parents):
    parents = [0]*num_parents
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999
    parents = np.array(parents)
    return parents

def crossover(parents, offspring_size):

    offspring = [0]*offspring_size[0]
    
    for k in range(offspring_size[0]): 
  
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)

            weights1=np.hstack((parents[parent1_idx].fc1.weight.data.numpy().flatten(), parents[parent1_idx].fc2.weight.data.numpy().flatten(), parents[parent1_idx].fc3.weight.data.numpy().flatten())).flatten()
            weights2=np.hstack((parents[parent2_idx].fc1.weight.data.numpy().flatten(), parents[parent2_idx].fc2.weight.data.numpy().flatten(), parents[parent2_idx].fc3.weight.data.numpy().flatten())).flatten()
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        weights1[j] = weights2[j]
                break
        offspring[k] = NeuralNetwork()
        offspring[k].fill_weights(weights1)
    offspring = np.array(offspring)
    return offspring


# flat curve result
# def mutation(offspring_crossover, mutation_intensity):
#     # mutating the offsprings generated from crossover to maintain variation in the population
#     num_genes_mutate = (int)(mutation_intensity*offspring_crossover.shape[1]/100)
#     for idx in range(offspring_crossover.shape[0]):
#         for _ in range(num_genes_mutate):
#             i = randint(0, offspring_crossover.shape[1]-1)
#             random_value = np.random.choice(np.arange(-1, 1, step = 0.001), size = (1), replace = False)
#             offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
#     return offspring_crossover


def mutation(offspring_crossover, mutation_intensity):

    size=offspring_crossover[0].fc1.weight.data.numel()+offspring_crossover[0].fc2.weight.data.numel()+offspring_crossover[0].fc3.weight.data.numel()
    for index in range(offspring_crossover.shape[0]):
        for i in range(size):
            if random.uniform(0, 1) < mutation_intensity:
                random_value = np.random.choice(np.arange(-1, 1, step = 0.01), size = (1), replace = False)
                offspring_crossover[index].update_individual_weights(i, random_value[0])
        # for i in range(offspring_crossover.shape[1]):
        #     if random.uniform(0, 1) < mutation_intensity:
        #         random_value = np.random.choice(np.arange(-1, 1, step = 0.001), size = (1), replace = False)
        #         offspring_crossover[index, i] = offspring_crossover[index, i] + random_value
    return offspring_crossover
