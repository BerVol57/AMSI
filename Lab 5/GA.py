import numpy as np
import random

class GA:
    def __init__(self, 
                        items: np.ndarray, 
                        max_weight: int, 
                 population_size=100, 
                 generations=1000, 
                 mutation_rate=0.1, 
                 tournament_size=3):


        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        
        # Список предметів (вага, вартість)
        self.items = items
        
        # Максимальна вага рюкзака
        self.max_weight = max_weight
        
        self.w = int(self.max_weight // min(self.items, key=lambda x: x[0])[0])
        self.w = self.w.bit_length()
        
        self.to_int = np.ones(self.w, dtype=int)
        for i in range(1, self.w):
            self.to_int[i] = 2*self.to_int[i-1]
        self.to_int = self.to_int[::-1]

    # Функція пристосованості
    def fitness(self, chromosome):
        a = self.to_int @ chromosome
        
        for i in range(len(a)):
            a[i] = np.round(a[i]/(1<<self.w) * (self.max_weight/self.items[i][0]))
            
        total_weight = self.items[..., 0] @ a
        total_value = self.items[..., 1] @ a
        # Штраф за перевищення ваги
        if total_weight > self.max_weight:
            return -1
        return total_value

    # Створення випадкової хромосоми
    def create_chromosome(self, length):
        chromosome = np.random.randint(0, 2, (self.w, length), dtype=int)
        return chromosome

    # Відбір турніром
    def tournament_selection(self, population):
        contestants = random.sample(population, self.tournament_size)
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]

    # Одноточковий кросовер
    def crossover(self, parent1, parent2):
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent1)
        
        for i in range(len(parent1)):
            point = random.randint(1, self.w)
            
            child1[i, :point] = parent1[i, :point]
            child1[i, point:] = parent2[i, point:]
            
            child2[i, :point] = parent1[i, :point]
            child2[i, point:] = parent2[i, point:]
            
            # child1[i] = ((parent1[i] >> point) << point)\
            #     ^ (parent2[i] & ((1 << point) - 1))
            
            # child2[i] = ((parent2[i] >> point) << point)\
            #     ^ (parent1[i] & ((1 << point) - 1))
            
            return child1, child2
        return parent1, parent2

    # Мутація
    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            for chrom in chromosome:
                if random.random() < self.mutation_rate:
                    a, b = sorted(np.random.choice(chromosome.shape[1], 2))
                    # Зміна значення хромосими
                    chrom[a:b] = chrom[a:b][::-1]
        return chromosome

    # Генетичний алгоритм
    def run(self):
        best_choises = []
        best_fitnesses = np.zeros(self.generations, dtype=int)
        
        population = []
        chromosome_length = self.items.shape[0]
        
        # Ініціалізація популяції
        for _ in range(self.population_size):
            chrom = self.create_chromosome(chromosome_length)
            fit = self.fitness(chrom)
            population.append((chrom, fit))
        
        for gen in range(self.generations):
            # Відбір та створення нового покоління
            new_population = []
            
            elite = max(population, key=lambda x: x[1])
            new_population.append(elite)
            
            while len(new_population) < self.population_size:
                # Вибір батьків
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # Схрещування
                child1, child2 = self.crossover(parent1, parent2)
                
                # Мутація
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Додавання дітей
                for child in [child1, child2]:
                    if len(new_population) >= self.population_size:
                        break
                    fit = self.fitness(child)
                    new_population.append((child, fit))
            
            population = new_population
            # Збереження найкращого результату
            current_best = max(population, key=lambda x: x[1])
            
            best_fitnesses[gen] = current_best[1]
            best_choises.append(current_best[0])
        
        
        return best_choises, best_fitnesses
