import random
from axo import Axo, axo_method

class ScatterSearch(Axo):
    def __init__(self, lower, upper,pop_size, refset_size,max_iter ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper
        self.pop_size = pop_size
        self.refset_size = refset_size
        self.max_iter = max_iter
        
        
    def objective(self ,x , **kwargs):
        return x ** 2

    def initialize_population(self,*args, **kwargs):
        return [random.uniform(self.lower, self.upper) for _ in range(self.pop_size)]

    def improve(self, solution, *args, **kwargs):
        delta = random.uniform(-0.1, 0.1)
        candidate = solution + delta
        candidate = max(self.lower, min(self.upper, candidate))
        return candidate if self.objective(candidate) < self.objective(solution) else solution

    def improve_population(self, population, *args, **kwargs):
        return [self.improve(sol) for sol in population]

    def update_refset(self, population, *args, **kwargs):
        sorted_pop = sorted(population, key=self.objective)
        return sorted_pop[:self.refset_size]

    def generate_subsets(self, refset, *args, **kwargs):
        return [(refset[i], refset[j]) for i in range(len(refset)) for j in range(i + 1, len(refset))]

    def combine(self, s1, s2, *args, **kwargs):
        return (s1 + s2) / 2
    @axo_method
    def scatter(self,show_plot:bool= False, *args, **kwargs):

        pop = self.initialize_population()
        pop = self.improve_population(pop)
        refset = self.update_refset(pop)

        fx_history = [self.objective(min(refset, key=self.objective))]

        for _ in range(self.max_iter):
            subsets = self.generate_subsets(refset)
            new_solutions = []

            for s1, s2 in subsets:
                combined = self.combine(s1, s2)
                improved = self.improve(combined)
                new_solutions.append(improved)

            refset += new_solutions
            refset = self.update_refset(refset)

            best_fx = self.objective(min(refset, key=self.objective))
            fx_history.append(best_fx)

        if show_plot: 
            self.plot(fx_history)

        return min(refset, key=self.objective),fx_history

    @axo_method
    async def plot(self, fx_history, *args, **kwargs):
        
        import matplotlib.pyplot as plt
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService
        
        sink_bucket_id      = kwargs.get("sink_bucket_id","test" )
        sink_key            = kwargs.get("sink_key", uuid4().hex)
        
        storage:MictlanXStorageService = kwargs.get("storage")
        
        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='purple')
        plt.title("Convergencia del Scatter Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id,key=sink_key,data=buf.getvalue())
        return sink_bucket_id,sink_key