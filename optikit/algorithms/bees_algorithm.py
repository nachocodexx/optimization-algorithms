import random
import matplotlib.pyplot as plt
import numpy as np
import os
from axo import Axo, axo_method

class BeesAlgorithm(Axo):
    def __init__(self,lower_bound: float = -100,upper_bound: float = 100,n: int = 10,m: int = 5,e: int = 2,nep: int = 3,nsp: int = 2,ngh: float = 1.0,max_iter: int = 50,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.lower = lower_bound
        self.upper = upper_bound
        self.n = n      
        self.m = m       
        self.e = e       
        self.nep = nep   
        self.nsp = nsp   
        self.ngh = ngh   
        self.max_iter = max_iter


    def objective(self, x, **kwargs):
        return x ** 2

    def _random_solution(self, **kwargs):
        return random.uniform(self.lower, self.upper)

    def _neighborhood(self, x, **kwargs):
        delta = random.uniform(-self.ngh, self.ngh)
        neighbor = x + delta
        return max(self.lower, min(neighbor, self.upper))

    @axo_method
    def run(self, show_plot: bool = False, **kwargs):
        population = [self._random_solution() for _ in range(self.n)]
        best_solution = min(population, key=self.objective)

        history_fx = []
        history_elite = []
        history_sites = []
        history_scouts = []

        for it in range(self.max_iter):
            population.sort(key=self.objective)
            new_population = []

            elite_sites = []
            selected_sites = []

            # Élite
            for i in range(self.e):
                patch_center = population[i]
                recruits = [self._neighborhood(patch_center) for _ in range(self.nep)]
                best_patch = min(recruits, key=self.objective)
                new_population.append(best_patch)
                elite_sites.append(best_patch)

            # Resto de sitios
            for i in range(self.e, self.m):
                patch_center = population[i]
                recruits = [self._neighborhood(patch_center) for _ in range(self.nsp)]
                best_patch = min(recruits, key=self.objective)
                new_population.append(best_patch)
                selected_sites.append(best_patch)

            # Exploradoras
            scouts = [self._random_solution() for _ in range(self.n - self.m)]
            new_population.extend(scouts)

            # Historial
            population = new_population
            best_candidate = min(population, key=self.objective)
            if self.objective(best_candidate) < self.objective(best_solution):
                best_solution = best_candidate

            history_fx.append(self.objective(best_solution))
            history_elite.append(elite_sites)
            history_sites.append(selected_sites)
            history_scouts.append(scouts)
            if show_plot: 
                self._plot_iteration(it, elite_sites, selected_sites, scouts)

        if show_plot:
            self._plot_convergence(history_fx)

        return best_solution

    def _plot_iteration(self, iter_num, elites, sites, scouts, **kwargs):
        os.makedirs("./img", exist_ok=True)
        x_curve = np.linspace(self.lower, self.upper, 400)
        y_curve = x_curve ** 2

        plt.figure(figsize=(8, 4))
        plt.plot(x_curve, y_curve, color='lightgray', label="f(x) = x²")
        plt.scatter(scouts, [self.objective(x) for x in scouts], color='skyblue', label='Exploradoras')
        plt.scatter(sites, [self.objective(x) for x in sites], color='orange', label='Sitios Seleccionados')
        plt.scatter(elites, [self.objective(x) for x in elites], color='crimson', label='Élite')
        plt.title(f"Iteración {iter_num + 1}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"img/bees_iter_{iter_num+1:02d}.png")
        plt.close()

    @axo_method
    async def _plot_convergence(self, fx_history, **kwargs):
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService

        sink_bucket_id = kwargs.get("sink_bucket_id", "test")
        sink_key = kwargs.get("sink_key", uuid4().hex)
        storage: MictlanXStorageService = kwargs.get("storage")

        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='green') 
        plt.title("Convergencia del Bees Algorithm")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id, key=sink_key, data=buf.getvalue())
        
        return sink_bucket_id, sink_key
