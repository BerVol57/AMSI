import numpy as np
import matplotlib.pyplot as plt

class TSPSolverACO:
    def __init__(self, cities, num_ants=50, iterations=200, 
                 alpha=.3, beta=1.0, rho=0.6, Q=100):

        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        
        
        self.dist_matrix = np.sqrt(
            ((cities[:, np.newaxis, :] - cities[np.newaxis, :, :])**2).sum(axis=2)
        )
        self.dist_matrix += np.eye(self.num_cities) * 1e-10  # Уникнення ділення на 0
        
        # Ініціалізація феромонів
        self.pheromone = np.ones((self.num_cities, self.num_cities)) 
        np.fill_diagonal(self.pheromone, 0)  # Немає феромонів на діагоналі
        
        # Збереження найкращого маршруту
        self.best_route = None
        self.best_distance = float('inf')
    
    def _construct_solutions(self):

        routes = np.zeros((self.num_ants, self.num_cities), dtype=int)
        for ant in range(self.num_ants):
            visited = np.zeros(self.num_cities, dtype=bool)
            current = np.random.randint(self.num_cities)
            routes[ant, 0] = current
            visited[current] = True
            
            for step in range(1, self.num_cities):
                # Обчислення ймовірностей переходу
                unvisited = ~visited
                pheromone = self.pheromone[current, unvisited] ** self.alpha
                heuristic = (1.0 / self.dist_matrix[current, unvisited]) ** self.beta
                probabilities = pheromone * heuristic
                probabilities /= probabilities.sum()
                
                # Вибір наступного міста
                next_city = np.random.choice(
                    np.where(unvisited)[0], 
                    p=probabilities
                )
                
                routes[ant, step] = next_city
                visited[next_city] = True
                current = next_city
        return routes
    
    def _update_pheromone(self, routes, distances):

        # Випаровування
        self.pheromone *= (1 - self.rho)
        
        # Додавання нового феромону
        for ant in range(self.num_ants):
            for i in range(self.num_cities-1):
                city_a = routes[ant, i]
                city_b = routes[ant, i+1]
                self.pheromone[city_a, city_b] += self.Q / distances[ant]
            # Замикання циклу
            city_a = routes[ant, -1]
            city_b = routes[ant, 0]
            self.pheromone[city_a, city_b] += self.Q / distances[ant]
    
    def run(self):

        history = []
        best_routes = []
        for iter in range(self.iterations):
            # Побудова маршрутів
            routes = self._construct_solutions()
            
            # Обчислення довжин маршрутів
            distances = np.zeros(self.num_ants)
            for ant in range(self.num_ants):
                dist = 0
                for i in range(self.num_cities-1):
                    dist += self.dist_matrix[routes[ant, i], routes[ant, i+1]]
                dist += self.dist_matrix[routes[ant, -1], routes[ant, 0]]
                distances[ant] = dist
                if dist < self.best_distance:
                    self.best_distance = dist
                    self.best_route = routes[ant].copy()
            
            # Оновлення феромонів
            self._update_pheromone(routes, distances)
            
            history.append(self.best_distance)
            if iter % 10 == 0:
                print(f"Iteration {iter}: Best Distance = {self.best_distance:.2f}")
                best_routes.append(self.best_route)
        
        return history, best_routes
    
    def plot_route(self):

        ordered_cities = self.cities[self.best_route]
        plt.figure(figsize=(8, 6))
        plt.scatter(ordered_cities[:, 0], ordered_cities[:, 1])
        plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], c="r")
        plt.plot([ordered_cities[-1, 0], ordered_cities[0, 0]],
                 [ordered_cities[-1, 1], ordered_cities[0, 1]], c="r")
        plt.title(f"Найкращий маршрут (Довжина: {self.best_distance:.2f})")
        plt.show()