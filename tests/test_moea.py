from optikit.algorithms.moead import MOEAD
from optikit.problems.problem import Problems
import pytest
from axo.contextmanager import AxoContextManager
import matplotlib.pyplot as plt
import pytest_asyncio
from axo.storage.services import MictlanXStorageService
from axo.endpoint.manager import DistributedEndpointManager
from axo import Axo
from optikit.algorithms.individual import Individual



problem = Problems()
fn_problem = problem.evaluate_zdt1



n_var = 30
bounds = [(0, 1)] * n_var
n_runs = 2
tiempos = []

indi_list = [Individual(n_var, bounds) for _ in range(100)]
for ind in indi_list:
    ind.evaluate(fn_problem)
    


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
async def test_local_moea():
    with AxoContextManager.local() as dcm:
        moead: MOEAD = MOEAD(fn_problem, n_var, bounds, n_gen=200, n_sub=100, T=20, population = indi_list, axo_endpoint_id="axo-endpoint-0")
        _ = await moead.persistify()
        result = moead.moea()
        assert result.is_ok
        pareto = moead.get_pareto_front()

@pytest.mark.asyncio
async def test_distributed_moea(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"
        
        moead: MOEAD = MOEAD(
         fn_problem,
         n_var,
         bounds,
         population = indi_list,
         n_gen              = 200,
         n_sub              = 100,
         T                  = 20, 
         axo_key            = axo_key,
         axo_bucket_id      = axo_bucket_id,
         axo_endpoint_id    = axo_endpoint_id
         )
    
        moead.append_dependency("matplotlib==3.9.2")
    
        persistify_result = await moead.persistify()
        assert persistify_result.is_ok
    
        res = moead.moea()
        assert res.is_ok
    
        resultados = res.unwrap()
    
        res_plot = moead.plot(
            sink_bucket_id = sink_bucket_id,
            sink_key       = sink_key,
        )
    
        assert res_plot.is_ok
    

@pytest.mark.asyncio
async def test_distributed_moea_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager, storage_service=storage_service) as dcm:
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"
        
        moead: MOEAD = MOEAD(
         fn_problem,
         n_var,
         bounds,
         n_gen              = 200,
         n_sub              = 100,
         T                  = 20, 
          population = indi_list,
         axo_key            = axo_key,
         axo_bucket_id      = axo_bucket_id,
         axo_endpoint_id    = axo_endpoint_id
         )
    
        moead.append_dependency("matplotlib==3.9.2")
    
        persistify_result = await moead.persistify()
        assert persistify_result.is_ok
    

        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_res.is_ok
        
        moead = ao_res.unwrap()
        
        res = moead.moea()
        assert res.is_ok
    
        resultados = res.unwrap()
    
        res_plot = moead.plot(
            sink_bucket_id = sink_bucket_id,
            sink_key       = sink_key,
        )
    
        assert res_plot.is_ok
    