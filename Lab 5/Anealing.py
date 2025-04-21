import numpy as np
import random


class SA:
    def __init__(self, 
                        items: np.ndarray, 
                        max_weight: int, 
                 initial_temp=1000, 
                 cooling_rate=0.995, 
                 min_temp=1, 
                 iterations=5000):
        
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations = iterations
        
        self.items = items
        self.max_weight = max_weight

    # Функція пристосованості (з штрафом за перевищення ваги)
    def fitness(self, solution):
        # Обчислюємо загальну вагу та вартість
        total_weight    = self.items[..., 0] @ solution
        total_value     = self.items[..., 1] @ solution

        # Штрафуємо за перевищення ваги
        if total_weight > self.max_weight:
            return 0
        return total_value

    # Генерація початкового рішення
    def initial_solution(self, items_length):
        sol = np.zeros(items_length, dtype=int)
        
        for i in range(items_length):
            sol[i] = random.randint(0, self.max_weight // self.items[i][0])
        
        return sol

    # Генерація сусіднього рішення
    def get_neighbor(self, current_solution):
        neighbor = current_solution.copy()
        # Змінюємо випадковий біт
        index = random.randint(0, len(neighbor)-1)
        i_max = self.max_weight // self.items[index][0]
        
        delta = random.randint(-2, 2)
        neighbor[index] += delta
        neighbor[index] = np.clip(neighbor[index], 0, i_max)
        return neighbor

    # Алгоритм імітації відпалу
    def run(self):
        current_solution = self.initial_solution(len(self.items))
        current_fitness = self.fitness(current_solution)
        
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        temp = self.initial_temp
        best_solutions = [best_solution]
        fitness_history = [best_fitness]
        
        for _ in range(self.iterations):
            # Генеруємо сусіда
            neighbor = self.get_neighbor(current_solution)
            neighbor_fitness = self.fitness(neighbor)
            
            # Обчислюємо різницю енергії
            delta = neighbor_fitness - current_fitness
            
            # Приймаємо рішення
            if delta > 0 or np.exp(delta / temp) > random.random():
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # Оновлюємо найкращий результат
                if current_fitness > best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            
            # Охолодження
            temp *= self.cooling_rate
            best_solutions.append(best_solution)
            fitness_history.append(best_fitness)
        
        return best_solution, fitness_history
