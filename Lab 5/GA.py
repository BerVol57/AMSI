import numpy as np
import random

class GA:
    def __init__(self, 
                        items: np.ndarray, 
                        max_weight: int, 
                population_size=100, 
                generations=1000, 
                mutation_rate=0.2, 
                tournament_size=5):


        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Список предметів (вага, вартість)
        self.items = items
        
        # Максимальна вага рюкзака
        self.max_weight = max_weight
        
        self.w = 4
        
        if self.items.shape[1] == 3:
            self.noi_limit = self.items[..., 2]
        else:
            self.noi_limit = np.zeros(self.items.shape[0])
            for i in range(self.items.shape[0]):
                self.noi_limit[i] = self.max_weight // self.items[i][0]

    # Функція пристосованості
    def fitness(self, chromosome):
        counts = np.round(chromosome/((1 << self.w) - 1)* self.noi_limit)
        total_weight = self.items[..., 0] @ counts
        total_value = self.items[..., 1] @ counts
        # Штраф за перевищення ваги
        if total_weight > self.max_weight:
            return -1
        return total_value

    # Створення випадкової хромосоми
    def create_chromosome(self, length):
        chromosome = np.zeros(length, dtype=int)
        return chromosome

    # Відбір турніром
    def tournament_selection(self, population):
        contestants = random.sample(population, self.tournament_size)
        winner = max(contestants, key=lambda x: self.fitness(x))
        return winner

    # Одноточковий кросовер
    def crossover(self, parent1, parent2):
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent1)

        for i in range(parent1.shape[0]):
            
            point = random.randint(1, self.w-1)
            
            child1[i] = ((parent1[i] >> point) << point)\
                ^ (parent2[i] & ((1 << point) - 1))
            
            child2[i] = ((parent2[i] >> point) << point)\
                ^ (parent1[i] & ((1 << point) - 1))
            
            return child1, child2

    # Мутація
    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            for _ in range(chromosome.shape[0]):
                # Обираємо випадковий елемент хромосоми
                index = np.random.randint(0, chromosome.shape[0])
                # Інвертуємо значення випадкового біта
                a, b = sorted(np.random.randint(0, self.w, 2), reverse=True)
                
                mutated_chromosome = chromosome[index]
                mutated_chromosome ^= ((1 << (a - b)) - 1) << b
                chromosome[index] = mutated_chromosome
        return chromosome

    # Генетичний алгоритм
    def run(self):
        best_choises = np.zeros((self.generations, self.items.shape[0]), dtype=int)
        best_fitnesses = np.zeros(self.generations, dtype=int)
        
        population = []
        chromosome_length = self.items.shape[0]
        
        # Ініціалізація популяції
        for i in range(self.population_size):
            chrom = self.create_chromosome(chromosome_length)
            population.append(chrom)
        
        pop_value_history = []
        
        for gen in range(self.generations):
            # Відбір та створення нового покоління
            while len(population) < 2*self.population_size:
                # Вибір батьків
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Схрещування
                child1, child2 = self.crossover(parent1, parent2)
                
                # Мутація
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Додавання дітей
                population.append(child1)
                population.append(child2)
                
            # Збереження найкращого результату
            population = sorted(population, key=lambda x: self.fitness(x), reverse=True)
            population = population[:self.population_size]
            
            current_best = population[0]
            
            best_fitnesses[gen] = self.fitness(current_best)
            best_choises[gen] = current_best
            pop_value_history.append([self.fitness(population[i]) for i in range(self.population_size)])
        
        
        return best_choises, best_fitnesses, pop_value_history