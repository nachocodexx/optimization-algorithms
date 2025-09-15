from optikit.algorithms.individual import Individual
#from optikit.algorithms.utils import 
import random
import matplotlib.pyplot as plt
from axo import Axo, axo_method


class MOEAD(Axo):
    def __init__(self, problem_func, n_var, bounds, n_gen=100, n_sub=100, T=20 ,population = 50 ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_func = problem_func
        self.n_var = n_var
        self.bounds = bounds
        self.n_gen = n_gen
        self.n_sub = n_sub
        self.T = T
        self.weights = MOEAD.init_weights(n_sub)
        self.neighbors = MOEAD.get_neighbors(self.weights, T)
        self.population = population

        self.z = [min([ind.f[i] for ind in self.population]) for i in range(2)]

    @axo_method
    def moea(self,show_plot:bool = False,**kwargs):
        for gen in range(self.n_gen):
            for i in range(self.n_sub):
                P = self.neighbors[i]
                p1, p2 = random.sample(P, 2)
                child = self.recombine(self.population[p1], self.population[p2])
                child.evaluate(self.problem_func)

                # Actualiza z*
                self.z = [min(self.z[j], child.f[j]) for j in range(2)]

                # Reemplazo
                for j in P:
                    f1 = MOEAD.scalarizing_chebyshev(child.f, self.weights[j], self.z)
                    f2 = MOEAD.scalarizing_chebyshev(self.population[j].f, self.weights[j], self.z)
                    if f1 < f2:
                        self.population[j] = child.copy()

    def recombine(self, ind1, ind2, **kwargs):
        child = Individual(self.n_var, self.bounds)
        child.x = [(x + y) / 2 for x, y in zip(ind1.x, ind2.x)]
        return child

    def get_pareto_front(self, **kwargs):
        return [ind.f for ind in self.population]
    
    
    @axo_method
    async def plot(self, **kwargs):
        
        import matplotlib.pyplot as plt
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService
        
        sink_bucket_id      = kwargs.get("sink_bucket_id","test" )
        sink_key            = kwargs.get("sink_key", uuid4().hex)
        
        storage:MictlanXStorageService = kwargs.get("storage")
        
        
        pareto = self.get_pareto_front()
        f1_vals = [f[0] for f in pareto]
        f2_vals = [f[1] for f in pareto]

        plt.scatter(f1_vals, f2_vals ,s=10)
        plt.xlabel("f1")
        plt.ylabel("f2")
        plt.title("Pareto Front")
        buf = io.BytesIO()
        plt.savefig(buf, format='png') 
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id,key=sink_key,data=buf.getvalue())
        return sink_bucket_id,sink_key
    
    @staticmethod
    def euclidean_dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

    @staticmethod
    def init_weights(n_subproblems):
        weights = []
        for i in range(n_subproblems):
            w = i / (n_subproblems - 1)
            weights.append([w, 1 - w])
        return weights

    @staticmethod
    def get_neighbors(weights, T):
        neighbors = []
        for i, w in enumerate(weights):
            dists = [(j, MOEAD.euclidean_dist(w, weights[j])) for j in range(len(weights))]
            dists.sort(key=lambda x: x[1])
            neighbors.append([j for j, _ in dists[:T]])
        return neighbors

    @staticmethod
    def scalarizing_chebyshev(f, weight, z):
        return max(weight[i] * abs(f[i] - z[i]) for i in range(len(f)))

