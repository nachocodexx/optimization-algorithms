import pytest
from optikit.algorithms.dijkstra import DijkstraAlgorithm
from axo.contextmanager import AxoContextManager
import networkx as nx
import pytest_asyncio
from axo.storage.services import MictlanXStorageService
from axo.endpoint.manager import DistributedEndpointManager
from axo import Axo


@pytest_asyncio.fixture(scope="session",autouse=True)
async def before_all_tests():
    ss = MictlanXStorageService(
        bucket_id   = "b1",
        protocol    = "http",
        routers_str = "mictlanx-router-0:localhost:60666"
    )
    bids = ["ao1bucket","ao2bucket"]
    for bid in bids:
        res = await ss.client.delete_bucket(bid)
        print(f"BUCKET [{bid}] was clean")
    yield

edges = [
    ('A', 'B', 1),
    ('B', 'C', 2),
    ('C', 'A', 2),   
    ('C', 'Z', 2)
]#Dijkstra solo acepta aristas con peso positivo, por lo que no se puede usar el grafo con aristas negativas



G = nx.DiGraph()
G.add_weighted_edges_from(edges)

@pytest.fixture()
def endpoint_manager()->DistributedEndpointManager:
    dem = DistributedEndpointManager() 
    dem.add_endpoint(
        endpoint_id="axo-endpoint-0",
        hostname="localhost",
        protocol="tcp",
        req_res_port=16667,
        pubsub_port=16666
    )
    return dem

@pytest.fixture()
def storage_service()->MictlanXStorageService:
    return MictlanXStorageService(
        protocol="http",
        routers_str="mictlanx-router-0:localhost:60666"
    )


@pytest.mark.asyncio
async def test_local_Local_search():
    with AxoContextManager.local() as dcm:
        ad:DijkstraAlgorithm = DijkstraAlgorithm(G, axo_endpoint_id="axo-endpoint-0")
        _ = await ad.persistify()
        res = ad.run('A', 'Z')
        assert res.is_ok
        cost, path = res.unwrap()
        print("Dijkstra:", path, "Coste:", cost)
        # ad.plot(path)
       
    


@pytest.mark.asyncio
async def test_distributed_dijkstra(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"

        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"

        ad:DijkstraAlgorithm = DijkstraAlgorithm(
            graph             = G,
            axo_key           = axo_key,
            axo_bucket_id     = axo_bucket_id,
            axo_endpoint_id   = axo_endpoint_id,
        )
        ad.append_dependency("networkx==3.2.1")
    
        persistify_result = await ad.persistify()
    
        assert persistify_result.is_ok
    
        res = ad.run('A', 'Z')
    
        assert res.is_ok
    
        _,path   = res.unwrap()
    
        res_plot = ad.plot(
            path,
            sink_bucket_id  = sink_bucket_id,
            sink_key        = sink_key,
            save_plot       = True
        )
    
        assert res_plot.is_ok
    
@pytest.mark.asyncio
async def test_distributed_dijkstra_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"

        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"

        ad:DijkstraAlgorithm = DijkstraAlgorithm(
        graph             = G,
        axo_key           = axo_key,
        axo_bucket_id     = axo_bucket_id,
        axo_endpoint_id   = axo_endpoint_id,
        )
        ad.append_dependency("networkx==3.2.1")
    
        persistify_result = await ad.persistify()
    
        assert persistify_result.is_ok
    
        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
    
        assert ao_res.is_ok
    
        ad = ao_res.unwrap()

        res = ad.run('A', 'Z')
    
        assert res.is_ok
    
        _,path   = res.unwrap()
    
        res_plot = ad.plot(
            path,
            sink_bucket_id  = sink_bucket_id,
            sink_key        = sink_key,
            save_plot       = True
        )
    
        assert res_plot.is_ok
    
    
    
   