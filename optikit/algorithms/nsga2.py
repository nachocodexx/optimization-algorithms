import numpy as np
from optikit.problems.problem import Problems
from axo import Axo, axo_method


class NSGA2(Axo):
    def __init__(self, runs , iters, m_objs, pop_size,verbose, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.runs = runs
        self.iters = iters
        self.m_objs = m_objs
        self.pop_size = pop_size
        self.verbose = verbose
        
    #se crea la poblacion aleatoriamente con las variables de decision 
    
    def create_initial_population(self, **kwargs):
        return np.random.rand(self.pop_size, 10)  
    
    #se evalua la poblacion con la funcion de evaluacion del problema, se utiliza la funcion dtlz1 para evaluar la poblacion
    #
    def evaluate(self, population, **kwargs ):
        X = np.array(population)
        objectives = np.array([Problems.dtlz1(X, m, self.m_objs) for m in range(self.m_objs)]).T
        return objectives

    #se seleccionan los padres para la reproduccion, se eligen dos padres aleatoriamente y se comparan sus objetivos
    # si el padre i es mejor que el padre j se selecciona el padre i, de lo contrario se selecciona el padre j
    #se repite el proceso hasta completar la poblacion de padres
    def select_parents(self, objectives , **kwargs):
        mating_pool = []
        for _ in range(self.pop_size):
            i, j = np.random.choice(self.pop_size, size=2, replace=False)
            if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                mating_pool.append(i)
            else:
                mating_pool.append(j)
        return mating_pool
    
    #se genera una nueva poblacion a partir de la poblacion actual y los padres seleccionados, se aplica el operador de cruce y mutacion
    #se generan dos hijos a partir de los padres seleccionados, se aplica el operador de crossover y mutation a cada hijo  
    def generate_new_population(self, population, mating_pool):
        new_population = np.copy(population)
        
        for i in range(0, self.pop_size, 2):
            parents = (population[mating_pool[i]], population[mating_pool[i+1]])
            child1, child2 = NSGA2.crossover(parents)
            new_population[i] = NSGA2.mutation(child1)
            new_population[i+1] = NSGA2.mutation(child2)
        
        return new_population
    
    #se aplica el operador de cruce y mutacion a los padres seleccionados, se generan dos hijos a partir de los padres seleccionados, se aplica el operador de crossover y mutation a cada hijo

    @staticmethod
    def crossover(parents, prob=1.0, eta=20):
        if np.random.rand() < prob:
            c1, c2 = parents
            u = np.random.rand(*c1.shape)
            beta = np.ones_like(u)
            for i in range(len(u)):
                if u[i] <= 0.5:
                    beta[i] = (2 * u[i]) ** (1 / (eta + 1))
                else:
                    beta[i] = (1 / (2 * (1 - u[i]))) ** (1 / (eta + 1))
        
            child1 = 0.5 * ((1 + beta) * c1 + (1 - beta) * c2)
            child2 = 0.5 * ((1 - beta) * c1 + (1 + beta) * c2)
            return child1, child2
        else:
            return parents

    #se aplica el operador de mutacion a los hijos generados, se aplica el operador de mutacion a cada hijo

    @staticmethod
    def mutation(child, prob=1.0, eta=15):
        if np.random.rand() < prob:
            u = np.random.rand(*child.shape)
            for i in range(len(child)):
                if u[i] < 0.5:
                    delta = (2 * u[i]) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1 / (eta + 1))
                child[i] += delta
        return child


    #ejecucin del algortimo 
    @axo_method
    def nsga(self ,show_plot:bool = False, **kwargs):
        population = self.create_initial_population()
        objectives = self.evaluate(population)
        
        for gen in range(self.iters):
            if self.verbose:
                print(f"Generacion {gen+1} completada")
            
            mating_pool = self.select_parents(objectives)
            population = self.generate_new_population(population, mating_pool)
            objectives = self.evaluate(population)
        
        return population, objectives




