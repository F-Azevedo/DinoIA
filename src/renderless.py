import pygame
import os
import random
import math
import time
from multiprocessing import Pool
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

GAME_MODE = "AI_MODE"

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus4.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]


class Dinosaur:
    X_POS = 90
    Y_POS = 330
    Y_POS_DUCK = 355
    JUMP_VEL = 17
    JUMP_GRAV = 1.1

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = 0
        self.jump_grav = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck and not self.dino_jump:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 20:
            self.step_index = 0

        if userInput == "K_UP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "K_DOWN" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif userInput == "K_DOWN":
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = True
        elif not (self.dino_jump or userInput == "K_DOWN"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 10]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_duck:
            self.jump_grav = self.JUMP_GRAV * 4
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel
            self.jump_vel -= self.jump_grav
        if self.dino_rect.y > self.Y_POS + 10:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL
            self.jump_grav = self.JUMP_GRAV
            self.dino_rect.y = self.Y_POS

    def draw(self, SCREEN):
        pass

    def getXY(self):
        return self.dino_rect.x, self.dino_rect.y


class Obstacle:
    def __init__(self, image, type):
        super().__init__()
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()

        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < - self.rect.width:
            obstacles.pop(0)

    def draw(self, SCREEN):
        pass

    def getXY(self):
        return self.rect.x, self.rect.y

    def getHeight(self):
        return y_pos_bg - self.rect.y

    def getType(self):
        return self.type


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 345


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 325


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)

        # High, middle or ground
        if random.randint(0, 3) == 0:
            self.rect.y = 345
        elif random.randint(0, 2) == 0:
            self.rect.y = 260
        else:
            self.rect.y = 300
        self.index = 0

    def draw(self, SCREEN):
        pass


class KeyClassifier:
    def __init__(self, state):
        self.state = state
        self.classifier = KNeighborsClassifier(n_neighbors=1)
        aux = []
        for i in state:
            if i[1] < 325:
                aux.append(1)
            # Precisa pular
            elif i[1] >= 325:
                aux.append(2)
        self.classifier.fit(state, aux)

    def keySelector(self, distance, obHeight, speed, obType):
        x = np.array([[distance, y_pos_bg - obHeight]])
        predict = self.classifier.predict(x)
        if predict == 0:
            return "K_NO"
        elif predict == 1:
            return "K_DOWN"
        elif predict == 2:
            return "K_UP"

    def updateState(self, state):
        self.state = state


def first(x):
    return x[0]


class KeySimplestClassifier(KeyClassifier):
    def __init__(self, state):
        self.state = state

    def keySelector(self, distance, obHeight, speed, obType):
        self.state = sorted(self.state, key=first)
        for s, d in self.state:
            if speed < s:
                limDist = d
                break
        if distance <= limDist:
            if isinstance(obType, Bird) and obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        return "K_NO"

    def updateState(self, state):
        self.state = state


def playerKeySelector():
    userInputArray = pygame.key.get_pressed()

    if userInputArray[pygame.K_UP]:
        return "K_UP"
    elif userInputArray[pygame.K_DOWN]:
        return "K_DOWN"
    else:
        return "K_NO"


def playGame(aiPlayer, seed):
    global game_speed, x_pos_bg, y_pos_bg, points, obstacles
    random.seed(seed)

    run = True
    clock = pygame.time.Clock()
    player = Dinosaur()
    game_speed = 10
    x_pos_bg = 0
    y_pos_bg = 383
    points = 0
    obstacles = []
    death_count = 0
    spawn_dist = 0

    def score():
        global points, game_speed
        points += 0.25
        if points % 100 == 0:
            game_speed += 1

    while run:

        distance = 1500
        obHeight = 0
        obType = 2
        if len(obstacles) != 0:
            xy = obstacles[0].getXY()
            distance = xy[0]
            obHeight = obstacles[0].getHeight()
            obType = obstacles[0]

        if GAME_MODE == "HUMAN_MODE":
            userInput = playerKeySelector()
        else:
            userInput = aiPlayer.keySelector(distance, obHeight, game_speed, obType)

        if len(obstacles) == 0 or obstacles[-1].getXY()[0] < spawn_dist:
            spawn_dist = random.randint(0, 670)
            if random.randint(0, 2) == 0:
                obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 5) == 5:
                obstacles.append(Bird(BIRD))

        player.update(userInput)

        for obstacle in list(obstacles):
            obstacle.update()

        score()

        for obstacle in obstacles:
            if player.dino_rect.colliderect(obstacle.rect):
                death_count += 1
                return points


# Change State Operator

def change_state(state, position, vs, vd):
    aux = state.copy()
    s, d = state[position]
    ns = s + vs
    nd = d + vd
    if ns < 15 or nd > 1000:
        return []
    return aux[:position] + [(ns, nd)] + aux[position + 1:]


# Neighborhood

def generate_neighborhood(state):
    neighborhood = []
    state_size = len(state)
    for i in range(state_size):
        ds = random.randint(1, 10)
        dd = random.randint(1, 100)
        new_states = [change_state(state, i, ds, 0), change_state(state, i, (-ds), 0), change_state(state, i, 0, dd),
                      change_state(state, i, 0, (-dd))]
        for s in new_states:
            if s:
                neighborhood.append(s)
    return neighborhood


# Gradiente Ascent
def gradient_ascent(state, max_time):
    start = time.process_time()
    res, max_value = manyPlaysResults(KeySimplestClassifier(state), 3)
    better = True
    end = 0
    while better and end - start <= max_time:
        neighborhood = generate_neighborhood(state)
        better = False
        for s in neighborhood:
            aiPlayer = KeySimplestClassifier(s)
            res, value = manyPlaysResults(aiPlayer, 3)
            if value > max_value:
                state = s
                max_value = value
                better = True
        end = time.process_time()
    return state, max_value


# MY IMPLEMENTATION
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


def genetic(pop_size, max_iter, cross_ratio, mut_ratio, max_time, elite_pct):
    global aiPlayer

    start = time.process_time()
    opt_state = []
    opt_value = 0

    pop = initial_population(pop_size)
    conv = convergent(pop)
    iter = 0
    end = 0

    while not conv and iter < max_iter and end - start <= max_time:
        print(f"Geração: {iter}, Feature Size: {FEATURE_SIZE}")
        print(f"Time: End - Start: {end-start}")
        val_pop = []
        for s in pop:
            aiPlayer = KeyClassifier(s)
            res, value = manyPlaysResults(aiPlayer, ROUNDS)
            val_pop.append(evaluate_population(value, s))
            if value > opt_value:
                f = open("best.txt", "a")
                f.write(f"NOVO BEST, geração {iter} Estado: {s}\n")
                f.write(f"\tTivemos resultado res: {res}, value: {value}\n")
                f.close()
                opt_state = s
                opt_value = value

        new_pop = elitism(val_pop, elite_pct)
        selected = selection(val_pop, pop_size - len(new_pop))
        crossed = crossover_step(selected, cross_ratio)
        mutated = mutation_step(crossed, mut_ratio)
        pop = new_pop + mutated
        conv = convergent(pop)
        iter += 1
        end = time.process_time()

    return opt_state, opt_value, iter, conv


def manyPlaysResults(aiPlayer, rounds):
    results = []
    with Pool(os.cpu_count() - 2) as p:
        results = p.starmap(playGame, zip([aiPlayer] * rounds, range(rounds)))
    npResults = np.asarray(results)
    return_value = npResults.mean()
    if npResults.shape[0] > 1:
        return_value -= npResults.std()
    return results, return_value


def run_flavio():
    global aiPlayer
    print("Resultado Flávio:")
    initial_state = [(15, 250), (18, 350), (20, 450), (1000, 550)]
    aiPlayer = KeySimplestClassifier(initial_state)
    #   best_state, best_value = gradient_ascent(initial_state, 5000)
    #   aiPlayer = KeySimplestClassifier(best_state)
    res, value = manyPlaysResults(aiPlayer, 30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)


def run_fernando(Population):
    global aiPlayer
    print("Resultado Fernando:")
    best_state = [[529, 300], [575, 325], [161, 300], [829, 325], [554, 260], [162, 325], [562, 300], [624, 325],
                  [451, 260], [833, 300], [553, 325]]
    #   best_state, best_value, iter, conv = genetic(Population, 200, 0.9, 0.1, 5000, 20)
    #   print(f"Best state: {best_state}, best_value: {best_value}, iter: {iter}, conv: {conv}\n")
    aiPlayer = KeyClassifier(best_state)
    res, value = manyPlaysResults(aiPlayer, 30)
    npRes = np.asarray(res)
    print(res, npRes.mean(), npRes.std(), value)


FEATURE_SIZE = 11
ROUNDS = 3


def main():
    run_flavio()
    run_fernando(100)


if __name__ == '__main__':
    main()
