from genetic_algorithm import *
import time

sol_per_pop = 50
num_generations = 100
crossover_percentage = 0.2
mutation_intensity = 0.01

datetimeCurr = str(time.strftime("%Y%m%d-%H%M%S"))
filename = "output"+datetimeCurr+".txt"
graphname = "graph"+datetimeCurr+".png"

num_parents_mating = (int)(crossover_percentage*sol_per_pop) 
num_weights = input_nodes*hidden_nodes_1 + hidden_nodes_1*hidden_nodes_2 + hidden_nodes_2*output_nodes
max_fitness = []

# Defining the population size.
pop_size = (sol_per_pop, num_weights)

#Creating the initial population.
new_population = np.array([NeuralNetwork() for _ in range(sol_per_pop)])
for neural_net in new_population:
    neural_net.fill_weights(np.random.choice(np.arange(-1, 1, step = 0.01), size = (num_weights), replace=True))
file1 = open(filename, "w")
file1.close()

for generation in range(num_generations):
    fitness = cal_pop_fitness(new_population, filename)
    max_fitness.append(np.max(fitness))
    

    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents, offspring_size = (pop_size[0] - parents.shape[0], num_weights))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = mutation(offspring_crossover, mutation_intensity)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0]] = parents
    new_population[parents.shape[0]:] = offspring_mutation
    
gen_count = list(range(1, num_generations+1))
