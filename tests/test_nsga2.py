from optikit.algorithms.nsga2 import NSGA2
import pytest
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
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


    
@pytest.mark.asyncio
async def test_local_nsga2():

    
    with AxoContextManager.local() as dcm:
        nsga2: NSGA2 = NSGA2(
            runs            = 2,
            iters           = 50,
            m_objs          = 2,
            pop_size        = 100,
            verbose         = True,
            axo_endpoint_id="axo-endpoint-0")
        _ = await nsga2.persistify()
        result = nsga2.nsga()
        assert result.is_ok
        population , objectives  = result.unwrap()
        print("Final Population:\n", population)
        print("Objectives:\n", objectives)
        
        
        
@pytest.mark.asyncio
async def test_distributed_nsga2(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
   
   with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"
        
        ns:NSGA2 = NSGA2(
            runs            = 2,
            iters           = 50,
            m_objs          = 2,
            pop_size        = 100,
            verbose         = True,
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )
        
        pesistify_result = await ns.persistify()
        
        assert pesistify_result.is_ok
        
        res = ns.nsga()
        assert res.is_ok
        
@pytest.mark.asyncio
async def test_distributed_nsga2_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
   
   with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"
        
        ns:NSGA2 = NSGA2(
            runs            = 2,
            iters           = 50,
            m_objs          = 2,
            pop_size        = 100,
            verbose         = True,
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )
        
        pesistify_result = await ns.persistify()
        
        assert pesistify_result.is_ok
        
        ao_result = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_result.is_ok
        
        ns = ao_result.unwrap()
        
        res = ns.nsga()
        assert res.is_ok
        