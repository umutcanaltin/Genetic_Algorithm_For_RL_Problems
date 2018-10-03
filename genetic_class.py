import numpy as np

class GeneticAgent:

    def __init__(self, population_size, chromosome_size, gene_variant,
                 elite_pop_size, mutation_chance, crossover_type, gym_env=True):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.gene_variant = gene_variant
        self.elite_pop_size = elite_pop_size
        self.mutation_chance = mutation_chance
        self.crossover_type = crossover_type
        self.gym_env = gym_env
        self.population = [np.random.randint(gene_variant, size=chromosome_size) for _ in range(population_size)]

    def select_elites(self, fitness_scores):
        population_ranks = list(reversed(np.argsort(fitness_scores)))
        return [self.population[x] for x in population_ranks[:self.elite_pop_size]]

    def select_parents(self, select_probs):

        return self.population[np.random.choice(range(self.population_size), p=select_probs)],\
            self.population[np.random.choice(range(self.population_size), p=select_probs)]

    def generate_new_population(self, fitness_scores):
        select_probs = np.array(fitness_scores) / np.sum(fitness_scores)
        child_set = []
        for generate in range(int((self.population_size-self.elite_pop_size)/2)):
            if self.crossover_type == "digit":
                children = self.cross_over_with_digit(self.select_parents(select_probs))
            else:
                children = self.cross_over_with_pieces(self.select_parents(select_probs))
            child_set.append(children[0])
            child_set.append(children[1])
        self.population = child_set + self.select_elites(fitness_scores=fitness_scores)

    def cross_over_with_digit(self, parents):
        children = {0: np.random.randint(self.gene_variant, size=self.chromosome_size),
                    1: np.random.randint(self.gene_variant, size=self.chromosome_size)}
        for child in range(2):
            for gene in range(self.chromosome_size):
                if np.random.uniform() < self.mutation_chance:
                    children[child][gene] = np.random.randint(4)
                else:
                    if np.random.uniform() < 0.5:
                        children[child][gene] = parents[0][gene]
                    else:
                        children[child][gene] = parents[1][gene]
        return children[0], children[1]

    def cross_over_with_pieces(self, parents):
        pos = np.random.randint(self.chromosome_size)
        child1 = np.concatenate((parents[0][:pos], parents[1][pos:]), axis=0)
        child2 = np.concatenate((parents[0][pos:], parents[1][:pos]), axis=0)
        for i in np.arange(self.chromosome_size):
            if np.random.uniform() < self.mutation_chance:
                child1[i] = np.random.randint(self.gene_variant)
        for i in np.arange(self.chromosome_size):
            if np.random.uniform() < self.mutation_chance:
                child2[i] = np.random.randint(self.gene_variant)
        return child1, child2

    def gym_play(self, env, total_game_number):
        fitness_scores = []
        if self.gym_env:

            for individual in self.population:
                total_reward = 0
                for ep in range(total_game_number):
                    state = env.reset()
                    while True:
                        # play from the policy (agent includes a action for every state)
                        action = individual[state]
                        next_state, reward, done, _ = env.step(action)
                        if done:
                            total_reward += reward
                            break
                        state = next_state
                fitness_scores.append(total_reward / total_game_number)
            return fitness_scores


