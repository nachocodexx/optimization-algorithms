from axo import Axo, axo_method

class DijkstraAlgorithm(Axo):
    def __init__(self, graph,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph

    @axo_method
    def run(self, source, target,show_plot:bool = False,**kwargs):
        import networkx as nx
        # Devuelve el camino y el costo total desde source hasta target
        path, cost = nx.single_source_dijkstra(self.graph, source=source, target=target)
        return path, cost

    @axo_method
    async def plot(self, path, algorithm_name="Dijkstra",save_plot:bool = False ,**kwargs):
        
        import networkx as nx
        import matplotlib.pyplot as plt
        import io
        from uuid import uuid4
        from axo.storage.services import MictlanXStorageService
        
        sink_bucket_id = kwargs.get("sink_bucket_id", "test") 
        sink_key       = kwargs.get("sink_key", uuid4().hex)
        
        storage:MictlanXStorageService = kwargs.get("storage")
        pos                            = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 5))
        nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color='lightblue')
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='crimson', width=3)
        if save_plot:
            plt.title(f"Camino más corto con {algorithm_name}")
            plt.axis('off')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            res = await storage.put(bucket_id=sink_bucket_id,key=sink_key,data=buf.getvalue())
        return sink_bucket_id,sink_key
