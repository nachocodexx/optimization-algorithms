from axo import Axo, axo_method
import random

class TabuSearch1D(Axo):
    def __init__(self, initial_solution=0, max_iter=50, tabu_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x0 = initial_solution
        self.max_iter = max_iter
        self.tabu_size = tabu_size
        
    def objective(self, x, **kwargs):
        return (x - 10) ** 2

    def neighbors(self, x, **kwargs):    
        return [x - 1, x + 1]
    
    @axo_method
    def tabu(self, show_plot: bool = False, *args, **kwargs):
        current = self.x0
        best = current
        tabu = []
        history = [self.objective(best)]

        for _ in range(self.max_iter):
            candidates = [s for s in self.neighbors(current) if s not in tabu]
            if not candidates:
                break
            candidate = min(candidates, key=self.objective)
            if self.objective(candidate) < self.objective(best):
                best = candidate
            tabu.append(candidate)
            if len(tabu) > self.tabu_size:
                tabu.pop(0)
            current = candidate
            history.append(self.objective(best))
        
        if show_plot: 
            self.plot(history)
        return best, self.objective(best), history

    @axo_method
    async def plot(self, history, *args, **kwargs):
        import matplotlib.pyplot as plt
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService

        sink_bucket_id = kwargs.get("sink_bucket_id", "test")
        sink_key = kwargs.get("sink_key", uuid4().hex)
        storage: MictlanXStorageService = kwargs.get("storage")
        
        plt.figure(figsize=(8, 4))
        plt.plot(history, marker='o', color='darkorange')
        plt.title("Convergencia de Tabú Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id, key=sink_key, data=buf.getvalue())
        return sink_bucket_id, sink_key
