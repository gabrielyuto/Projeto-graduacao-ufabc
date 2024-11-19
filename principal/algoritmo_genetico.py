import random

POPULATION_SIZE = 10
GENERATIONS = 100
MUTATION_RATE = 0.1
CHROMOSOME_LENGTH = 8

def fitness(chromosome):
    x = int("".join(map(str, chromosome)), 2)
    return x * x

def create_individual():
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


def select_pair(population):
    weights = [fitness(individual) for individual in population]
    return random.choices(population, weights=weights, k=2)


def crossover(parent1, parent2):
    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(individual):
    if random.random() < MUTATION_RATE:
        point = random.randint(0, CHROMOSOME_LENGTH - 1)
        individual[point] = 1 - individual[point]


def genetic_algorithm():
    population = create_population()

    for generation in range(GENERATIONS):
        new_population = []

        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_pair(population)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = new_population

        best_individual = max(population, key=fitness)
        best_fitness = fitness(best_individual)
        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")

    best_individual = max(population, key=fitness)
    best_fitness = fitness(best_individual)
    print(f"\nBest solution: x = {int(''.join(map(str, best_individual)), 2)}, f(x) = {best_fitness}")

