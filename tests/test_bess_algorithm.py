from optikit.algorithms.bees_algorithm import BeesAlgorithm
import pytest
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
import pytest_asyncio
from axo import Axo
from axo.storage.services import MictlanXStorageService


@pytest_asyncio.fixture(scope="session", autouse=True)
async def before_all_test():
    ss = MictlanXStorageService(
        bucket_id   = "b1",
        protocol    = "http",
        routers_str = "mitclanx-router-0:localhost:60666"
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
async def test_local_bess_algorithm():
    with AxoContextManager.local() as dcm:
        ba: BeesAlgorithm =  BeesAlgorithm(
            lower_bound     = -10,
            upper_bound     = 10,
            n               = 20,
            m               = 10,
            e               = 5,
            nep             = 3,
            nsp             = 2,
            ngh             = 0.5,
            max_iter        = 50,
            axo_endpoint_id="axo-endpoint-0"
        )
        _ = await ba.persistify()
        best_solution = ba.run()
        assert best_solution.is_ok
        print("Best solution found:", best_solution.unwrap())
        
@pytest.mark.asyncio 
async def test_distributed_bess_algorithm(endpoint_manager:DistributedEndpointManager, storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"

        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"
        ba: BeesAlgorithm =  BeesAlgorithm(
            lower_bound     = -10,
            upper_bound     = 10,
            n               = 20,
            m               = 10,
            e               = 5,
            nep             = 3,
            nsp             = 2,
            ngh             = 0.5,
            max_iter        = 50,
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )
        
        ba.append_dependency("matplotlib==3.9.2")
        
        persistify_result = await ba.persistify()
        
        assert persistify_result.is_ok
        
        res = ba.run()
        assert res.is_ok
        
        best_solution = res.unwrap()
        
        res_plot = ba._plot_convergence(
            best_solution,
            sink_bucket_id = sink_bucket_id,
            sink_key       = sink_key,
            save_plot      = True
        )
        
        assert res_plot.is_ok
    


@pytest.mark.asyncio 
async def test_distributed_bess_algorithm_zk(endpoint_manager:DistributedEndpointManager, storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:

        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"

        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"
        ba: BeesAlgorithm =  BeesAlgorithm(
            lower_bound     = -10,
            upper_bound     = 10,
            n               = 20,
            m               = 10,
            e               = 5,
            nep             = 3,
            nsp             = 2,
            ngh             = 0.5,
            max_iter        = 50,
            axo_key         = axo_key,
            axo_bucket_id   = axo_bucket_id,
            axo_endpoint_id = axo_endpoint_id
        )
        
        ba.append_dependency("matplotlib==3.9.2")
        
        persistify_result = await ba.persistify()
        
        assert persistify_result.is_ok
        
        
        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_res.is_ok
        
        ba = ao_res.unwrap()
        
        res = ba.run()
        assert res.is_ok
        
        best_solution = res.unwrap()
        
        res_plot = ba._plot_convergence(
            best_solution,
            sink_bucket_id = sink_bucket_id,
            sink_key       = sink_key,
            save_plot      = True
        )
        
        assert res_plot.is_ok