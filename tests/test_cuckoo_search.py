from optikit.algorithms.cuckoo_search import CuckooSearch
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
async def test_local_cuckoo_search():
    with AxoContextManager.local() as dcm:
        cs: CuckooSearch =  CuckooSearch(
            lower           = -10,
            upper           = 10,
            n               = 5,
            pa              = 0.5,
            max_iter        = 10,
            alpha           = 1.0,            
            axo_endpoint_id="axo-endpoint-0"
        )
        _ = await cs.persistify()
        res = cs.cuckoo()
        assert res.is_ok
        best = res.unwrap()
        print("Best solution found:", best)
        

@pytest.mark.asyncio
async def test_distributed_cuckoo_search(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):

    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucketd_id         = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"      
        
        cs:CuckooSearch = CuckooSearch (
            lower           = -10,
            upper           = 10,
            n               = 5,
            pa              = 0.5,
            max_iter        = 10,
            alpha           = 1.0,            
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )         
        
        cs.append_dependency("numpy==2.2.4")
        cs.append_dependency("matplotlib==3.9.2")
        
        persistify_result = await cs.persistify()
        
        print("RESULT DE PERSISTENCIA", persistify_result)
        assert persistify_result.is_ok
        
        res = cs.cuckoo()
        assert res.is_ok
        
        fx_history  = res.unwrap()
        
        res_plot    = cs.plot_convergence(
            fx_history,
            sink_bucketd_id = sink_bucketd_id,
            sink_key        = sink_key,
            save_plot       = True
        ) 
        assert res_plot.is_ok
        
@pytest.mark.asyncio
async def test_distributed_cuckoo_search_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):

    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucketd_id         = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"      
        
        cs:CuckooSearch = CuckooSearch (
            lower           = -10,
            upper           = 10,
            n               = 5,
            pa              = 0.5,
            max_iter        = 10,
            alpha           = 1.0,            
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )         
        
       
        
        persistify_result = await cs.persistify()
        
        print("RESULT DE PERSISTENCIA", persistify_result)
        
        assert persistify_result.is_ok
        
        print("BUCKET:",axo_bucket_id)
        
        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        print("RESULT:",ao_res)
        assert ao_res.is_ok
        
        cs = ao_res.unwrap()
        
        res = cs.cuckoo()
        
        assert res.is_ok
        
        fx_history  = res.unwrap()
        
        res_plot    = cs.plot_convergence(
            fx_history,
            sink_bucketd_id = sink_bucketd_id,
            sink_key        = sink_key,
            save_plot       = True
        ) 
        assert res_plot.is_ok
        
        
        