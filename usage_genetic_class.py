import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from genetic_class import GeneticAgent

env = gym.make("FrozenLake-v0")
genetic_agent = GeneticAgent(population_size=60, chromosome_size=env.observation_space.n, gene_variant=env.action_space.n,
                             elite_pop_size=4, mutation_chance=0.01, crossover_type="digit", gym_env=True)

best_scores_with_digit_crossover = []
best_scores_with_peaces_crossover = []
total_game_number = 100
total_generation_number = 100

for generation in range(total_generation_number):
    fitness_scores = genetic_agent.gym_play(env, total_game_number=total_game_number)
    genetic_agent.generate_new_population(fitness_scores)
    generation_score = np.average(fitness_scores)
    best_scores_with_digit_crossover.append(generation_score)


genetic_agent = GeneticAgent(population_size=20, chromosome_size=env.observation_space.n, gene_variant=env.action_space.n,
                             elite_pop_size=6, mutation_chance=0.01, crossover_type="piece", gym_env=True)
fitness_scores = []
for generation in range(total_generation_number):
    fitness_scores = genetic_agent.gym_play(env, total_game_number=total_game_number)
    generation_score = np.average(fitness_scores)
    best_scores_with_peaces_crossover.append(generation_score)
    genetic_agent.generate_new_population(fitness_scores)

y1 = best_scores_with_digit_crossover
y2 = best_scores_with_peaces_crossover

x = np.arange(total_generation_number)

plt.subplot(2, 1, 1)
plt.plot(x, y1, 'o-')
plt.title('Genetic Algorithms For RL')
plt.ylabel('score of digit_CO')
locator = ticker.MultipleLocator(2)
plt.gca().xaxis.set_major_locator(locator)
plt.subplot(2, 1, 2)
plt.plot(x, y2, '.-')
plt.xlabel('generation')
plt.ylabel('score of peace_CO')
plt.gca().xaxis.set_major_locator(locator)
plt.show()
