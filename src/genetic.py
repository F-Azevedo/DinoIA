import random
import math


FEATURE_SIZE = 11


def generate_initial_state():
    opt = [260, 300, 325, 345]
    initial_state = []
    for i in range(FEATURE_SIZE):
        initial_state.append([random.randint(0, 1100), opt[random.randint(0, 2)]])

    return initial_state


def initial_population(n):
    pop = []
    count = 0
    while count < n:
        individual = generate_initial_state()
        pop = pop + [individual]
        count += 1
    return pop


def convergent(population):
    if population:
        base = population[0]
        i = 0
        while i < len(population):
            if base != population[i]:
                return False
            i += 1
        return True
    return False


def evaluate_population(res, s):
    return [res, s]


def elitism(val_pop, pct):
    n = math.floor((pct / 100) * len(val_pop))
    if n < 1:
        n = 1
    val_elite = sorted(val_pop, key=lambda x: x[0], reverse=True)[:n]
    elite = [s for v, s in val_elite]
    return elite


def states_total_value(states):
    total_sum = 0
    for state in states:
        total_sum = total_sum + state[0]
    return total_sum


def roulette_construction(states):
    aux_states = []
    roulette = []
    total_value = states_total_value(states)

    for state in states:
        value = state[0]
        if total_value != 0:
            ratio = value / total_value
        else:
            ratio = 1
        aux_states.append((ratio, state[1]))

    acc_value = 0
    for state in aux_states:
        acc_value = acc_value + state[0]
        s = (acc_value, state[1])
        roulette.append(s)
    return roulette


def roulette_run(rounds, roulette):
    if not roulette:
        return []
    selected = []
    while len(selected) < rounds:
        r = random.uniform(0, 1)
        for state in roulette:
            if r <= state[0]:
                selected.append(state[1])
                break
    return selected


def selection(value_population, n):
    aux_population = roulette_construction(value_population)
    # print(f"Roulette : {aux_population}")
    new_population = roulette_run(n, aux_population)
    return new_population


def crossover(dad, mom):
    r = random.randint(0, len(dad) - 1)
    son = dad[:r] + mom[r:]
    daug = mom[:r] + dad[r:]

    return son, daug


def crossover_step(population, crossover_ratio):
    new_pop = []

    for _ in range(round(len(population) / 2)):
        rand = random.uniform(0, 1)
        fst_ind = random.randint(0, len(population) - 1)
        scd_ind = random.randint(0, len(population) - 1)
        parent1 = population[fst_ind]
        parent2 = population[scd_ind]

        if rand <= crossover_ratio:
            offspring1, offspring2 = crossover(parent1, parent2)
        else:
            offspring1, offspring2 = parent1, parent2

        new_pop = new_pop + [offspring1, offspring2]

    return new_pop


def mutation(indiv):
    individual = indiv.copy()
    rand = random.randint(0, len(individual) - 1)

    if individual[rand]:
        # ou só trocar aqui o individual[rand][0] para um random.randint(0, 1100)
        r = random.uniform(0, 1)
        if r > 0.5:
            individual[rand][0] = individual[rand][0] + 20
        else:
            individual[rand][0] = individual[rand][0] - 20

    return individual


def mutation_step(population, mutation_ratio):
    ind = 0
    for individual in population:
        rand = random.uniform(0, 1)

        if rand <= mutation_ratio:
            mutated = mutation(individual)
            population[ind] = mutated
        ind += 1

    return population
