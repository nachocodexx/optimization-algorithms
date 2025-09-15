
from scipy.optimize import minimize
from axo import Axo, axo_method


class SumaPonderada(Axo):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    # Función objetivo 1
    def f1(self, x, *args, **kwargs):
        return x[0]**2 + x[1]**2
    # Función objetivo 2
    def f2(self, x, *args, **kwargs):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    # Función objetivo combinada con pesos
    def weighted_sum_objective(self, x, weights, *args, **kwargs):
        return weights[0]*self.f1(x) + weights[1]*self.f2(x)

    def weighted_sum_method(self, weights, x0=[0, 0]  , *args, **kwargs):
        result = minimize(self.weighted_sum_objective, x0, args=(weights,))
        return result.x, result.fun

    @axo_method
    def suma(self, show_plot:bool = False,pasos=50, *args, **kwargs):
        
        import numpy as np
        
        solutions = []
        weights_list = []

        for alpha in np.linspace(0, 1, pasos):
            w = [alpha, 1 - alpha]
            x_opt, _ = self.weighted_sum_method(w)
            f1_val = self.f1(x_opt)
            f2_val = self.f2(x_opt)
            solutions.append([f1_val, f2_val])
            weights_list.append(w)

        return np.array(solutions), weights_list
    
    @axo_method
    async def plot(self, soluciones, *args, **kwargs):
        import matplotlib.pyplot as plt
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService
        
        sink_bucket_id      = kwargs.get("sink_bucket_id", "test")
        sink_key            = kwargs.get("sink_key", uuid4().hex)
        
        storage:MictlanXStorageService = kwargs.get("storage")
        
        plt.figure(figsize=(8, 6))
        plt.plot(soluciones[:, 0], soluciones[:, 1], 'bo-', label='Frente de Pareto (estimado)')
        plt.xlabel('f1(x)')
        plt.ylabel('f2(x)')
        plt.title('Frente de Pareto aproximado por suma ponderada')
        plt.grid(True)
        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id,key=sink_key,data=buf.getvalue())
        return sink_bucket_id,sink_key
    