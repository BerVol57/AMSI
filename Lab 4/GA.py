import matplotlib.pyplot as plt
import numpy as np

class TSPSolverGA:
    def __init__(self, cities, population_size=100, generations=500, 
                 mutation_rate=0.02, tournament_size=5):

        self.cities = cities
        self.num_cities = len(cities)
        self.pop_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Попередній розрахунок матриці відстаней
        self.dist_matrix = np.sqrt(((cities[:, np.newaxis, :] - cities[np.newaxis, :, :])**2 ).sum(axis=2))
        
        # Ініціалізація популяції
        self.population = np.array([np.random.permutation(self.num_cities) 
                                  for _ in range(self.pop_size)])
    
    def calculate_fitness(self, population):

        fitness = np.zeros(len(population))
        for i, route in enumerate(population):
            total_dist = self.dist_matrix[route[:-1], route[1:]].sum()
            # Повернення в початок
            total_dist += self.dist_matrix[route[-1], route[0]]
            fitness[i] = 1 / total_dist
        return fitness
    
    def tournament_selection(self, fitness):

        selected = np.zeros(self.pop_size, dtype=int)
        for i in range(self.pop_size):
            candidates = np.random.choice(len(fitness), self.tournament_size, replace=False)
            selected[i] = candidates[np.argmax(fitness[candidates])]
        return selected
    
    def ordered_crossover(self, parent1, parent2):

        size = len(parent1)
        a, b = sorted(np.random.choice(size, 2, replace=False))
        child = -np.ones(size, dtype=int)
        child[a:b] = parent1[a:b]
        
        remaining = [item for item in parent2 if item not in child[a:b]]
        ptr = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = remaining[ptr]
                ptr += 1
        return child
    
    def mutate(self, route):

        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(len(route), 2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route
    
    def evolve(self):

        best_positions = []
        best_fitness = []
        for gen in range(self.generations):
            # Оцінка пристосованості
            fitness = self.calculate_fitness(self.population)
            
            # Вибір батьків
            selected = self.tournament_selection(fitness)
            
            # Створення нової популяції
            new_pop = []
            for i in range(0, self.pop_size, 2):
                p1 = self.population[selected[i]]
                p2 = self.population[selected[i+1]]
                
                # Схрещування
                child1 = self.ordered_crossover(p1, p2)
                child2 = self.ordered_crossover(p2, p1)
                
                # Мутація
                new_pop.extend([self.mutate(child1), self.mutate(child2)])
            
            
            # Запис найкращого результату
            best_idx = np.argmax(fitness)
            best_fitness.append(1/fitness[best_idx])
            
            new_pop[0] = self.population[best_idx]
            self.population = np.array(new_pop)
            
            
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Distance = {best_fitness[-1]:.2f}")
                best_positions.append(self.population[0])
        
        return best_fitness, np.array(best_positions)
    
    def get_best_route(self):

        fitness = self.calculate_fitness(self.population)
        return self.population[np.argmax(fitness)]
    
    def plot_route(self):

        ordered_cities = self.cities[self.population[0]]
        plt.figure(figsize=(8, 6))
        plt.scatter(ordered_cities[:, 0], ordered_cities[:, 1])
        plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], c="r")
        plt.plot([ordered_cities[-1, 0], ordered_cities[0, 0]],
                    [ordered_cities[-1, 1], ordered_cities[0, 1]], 'r-', c="r")
        plt.title("Найкращий маршрут")
        plt.tight_layout()