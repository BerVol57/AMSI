import numpy as np
import random

class GA:
    def __init__(self, 
                        items: np.ndarray, 
                        max_weight: int, 
                 population_size=100, 
                 generations=1000, 
                 mutation_rate=0.2, 
                 tournament_size=3, 
                 elitism=True):


        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        # Використовувати елітизм чи ні
        self.elitism = elitism
        
        # Список предметів (вага, вартість)
        self.items = items
        
        self.min_weight = min(self.items, key=lambda x: x[0])[0]
        
        # Максимальна вага рюкзака
        self.max_weight = max_weight

    # Функція пристосованості
    def fitness(self, chromosome):
        total_weight = self.items[..., 0] @ chromosome
        total_value = self.items[..., 1] @ chromosome
        # Штраф за перевищення ваги
        if total_weight > self.max_weight:
            return 0
        return total_value

    # Створення випадкової хромосоми
    def create_chromosome(self, length):
        chromosome = np.zeros(length, dtype=int)
        for i in range(length):
            chromosome[i] = random.randint(0, self.max_weight//self.items[i][0])
        return chromosome

    # Відбір турніром
    def tournament_selection(self, population):
        selected = []
        # Обираємо 2-ох батьків з турніру
        contestants = random.sample(population, 2*self.tournament_size)
        winners = sorted(contestants, key=lambda x: x[1])
        
        for i in range(2):
            selected.append(winners[i][0])
        return selected

    # Одноточковий кросовер
    def crossover(self, parent1, parent2):
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent1)
        
        for i in range(len(parent1)):
            min_blength = self.max_weight.bit_length() + 1
            
            point = random.randint(1, min_blength-1)
            
            child1[i] = ((parent1[i] >> point) << point)\
                ^ (parent2[i] & ((1 << point) - 1))
            
            child2[i] = ((parent2[i] >> point) << point)\
                ^ (parent1[i] & ((1 << point) - 1))
            
            return child1, child2
        return parent1, parent2

    # Мутація
    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                # Зміна значення хромосими
                i_max = self.max_weight//self.items[i][0]
                mutated_chromosome = i_max - chromosome[i]
                mutated_chromosome = np.clip(mutated_chromosome, 0, i_max)
                chromosome[i] = mutated_chromosome
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
            
            if self.elitism:
                # Елітизм: зберегти найкращу хромосому
                elite = max(population, key=lambda x: x[1])
                new_population.append(elite)
            
            while len(new_population) < self.population_size:
                # Вибір батьків
                parents = self.tournament_selection(population)
                
                # Схрещування
                child1, child2 = self.crossover(parents[0], parents[1])
                
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
