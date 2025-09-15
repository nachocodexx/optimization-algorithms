
from axo import Axo, axo_method


class LocalSearch(Axo):
    def __init__(self, x, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = x
        self.trayectoria = []  # historial de soluciones
        self.fx = []     

    def obj_function(self, x):
        return x**2

    def generate_neighbors(self, x , **kwargs):
        return [x - 1, x + 1]

    def conditional(self, x, **kwargs):
        return abs(x) == 0  
    
    @axo_method
    def local(self, **kwargs):
        s = self.x
        self.trayectoria.append(s)
        self.fx.append(self.obj_function(s))
        print("Punto de partida:", s)
        print("Valor objetivo inicial:", self.obj_function(s))
        
        while not self.conditional(s):
            neighbors = self.generate_neighbors(s)
            print("Vecinos:", neighbors)

            best_neighbor = None
            best_value = float('inf')

            for neighbor in neighbors:
                value = self.obj_function(neighbor)
                if value < best_value:
                    best_value = value
                    best_neighbor = neighbor

            if best_value < self.obj_function(s):
                s = best_neighbor
                self.trayectoria.append(s)
                self.fx.append(self.obj_function(s))
                
            else:
                break

        print("Mejor solución encontrada:", s)
        return s
    
    @axo_method
    async def plot(self, **kwargs):
        import io 
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService 
        import matplotlib.pyplot as plt
        
        sink_bucket_id      = kwargs.get("sink_bucket_id", "test")
        sink_key            = kwargs.get("sink_key",uuid4().hex)
        
        storage:MictlanXStorageService = kwargs.get("storage")
        
        plt.figure(figsize=(8, 4))
        plt.plot(self.fx, marker='o', color='teal')
        plt.title("Convergencia de Búsqueda Local en $f(x) = x^2$")
        plt.xlabel("Iteración")
        plt.ylabel("f(x)")
        plt.grid()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        res = await storage.put(bucket_id=sink_bucket_id,key=sink_key,data=buf.getvalue())
        return sink_bucket_id,sink_key
