import pytest
from optikit.algorithms.local_search import LocalSearch
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import matplotlib.pyplot as plt
import pytest_asyncio
from axo.storage.services import MictlanXStorageService
from axo import Axo

@pytest_asyncio.fixture(scope="session", autouse=True)
# @pytest.mark.asyncio
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

@pytest.fixture()
def dem():
    dem = DistributedEndpointManager()
    dem.add_endpoint(
        endpoint_id  = "axo-endpoint-0",
        hostname     = "localhost",
        protocol     = "tcp",
        req_res_port = 16667,
        pubsub_port  = 16666
    )
    return dem

@pytest.mark.asyncio
async def test_local_Local_search():
    with AxoContextManager.local() as dcm:
        ls:LocalSearch = LocalSearch(x=50, axo_endpoint_id="axo-endpoint-0")
        _ = await ls.persistify()
        res = ls.local()
        assert res.is_ok
        mejor_solucion = res.unwrap()
        
        print("Mejor solución encontrada:", mejor_solucion)
        
@pytest.mark.asyncio
async def test_distributed_local_search(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"
        
        ls:LocalSearch = LocalSearch(
            x=50, 
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
            )
        
        ls.append_dependency("matplotlib==3.9.2")
        
        persistify_result = await ls.persistify()
        assert persistify_result.is_ok
        
        res = ls.local()
        assert res.is_ok
        
        mejor_solucion = res.unwrap()
        
        print("Mejor solución encontrada:", mejor_solucion)
        
        res_plot = ls.plot(
            sink_bucket_id = sink_bucket_id,
            sink_key        = sink_key,
        )
        
        assert res_plot.is_ok
        

@pytest.mark.asyncio
async def test_distributed_local_search_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"
        
        ls:LocalSearch = LocalSearch(
            x=50, 
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
            )
        
        ls.append_dependency("matplotlib==3.9.2")
        
        persistify_result = await ls.persistify()
        assert persistify_result.is_ok
        
        ao_result = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_result.is_ok
        ls = ao_result.unwrap()
            
        res = ls.local()
        assert res.is_ok
        
        mejor_solucion = res.unwrap()
        
        print("Mejor solución encontrada:", mejor_solucion)
        
        res_plot = ls.plot(
            sink_bucket_id = sink_bucket_id,
            sink_key        = sink_key,
        )
        
        assert res_plot.is_ok
        
        
        
        