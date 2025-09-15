import numpy as np
import math
from axo import Axo, axo_method

import numpy as np
import math
from axo import Axo, axo_method

class CuckooSearch(Axo):
    def __init__(self, lower, upper, n, pa, max_iter, alpha=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper
        self.n = n                    # Número de nidos
        self.pa = pa                  # Probabilidad de descubrimiento
        self.max_iter = max_iter
        self.alpha = alpha            # Escala del vuelo Lévy

    def objective(self, x):
        return x ** 2

    def levy_flight(self,**kwargs):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return self.alpha * step

    def simple_bounds(self, x ,**kwargs):
        return max(self.lower, min(x, self.upper))
    @axo_method
    def cuckoo(self, show_plots: bool = False, **kwargs):
        import random

        nests = [random.uniform(self.lower, self.upper) for _ in range(self.n)]
        fitness = [self.objective(x) for x in nests]
        best = nests[np.argmin(fitness)]
        fx_history = [self.objective(best)]

        for _ in range(self.max_iter):
            for i in range(self.n):
                x = nests[i]
                step = self.levy_flight()
                new_x = self.simple_bounds(x + step)
                new_f = self.objective(new_x)

                j = random.randint(0, self.n - 1)
                if new_f < fitness[j]:
                    nests[j] = new_x
                    fitness[j] = new_f

            for i in range(self.n):
                if random.random() < self.pa:
                    nests[i] = random.uniform(self.lower, self.upper)
                    fitness[i] = self.objective(nests[i])

            current_best = nests[np.argmin(fitness)]
            if self.objective(current_best) < self.objective(best):
                best = current_best

            fx_history.append(self.objective(best))

        if show_plots: 
            self.plot_convergence(fx_history)

        return best, fx_history 

    @axo_method
    async def plot_convergence(self, fx_history, **kwargs):
        import matplotlib.pyplot as plt
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService
        import io

        sink_bucket_id = kwargs.get("sink_bucket_id", "test")
        sink_key = kwargs.get("sink_key", uuid4().hex)
        storage: MictlanXStorageService = kwargs.get("storage")

        plt.figure(figsize=(8, 4))
        plt.plot(fx_history, marker='o', color='navy')
        plt.title("Convergencia de Cuckoo Search")
        plt.xlabel("Iteración")
        plt.ylabel("Mejor f(x)")
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id, key=sink_key, data=buf.getvalue())
        return sink_bucket_id, sink_key
