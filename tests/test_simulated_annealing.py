import pytest
from optikit.algorithms.simulated_annealing import SimulatedAnnealing
import random
import matplotlib.pyplot as plt
from axo.contextmanager import AxoContextManager
from axo.endpoint.manager import DistributedEndpointManager
from axo import Axo
import pytest_asyncio
from axo.storage.services import MictlanXStorageService


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
async def test_simulated_annealing(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"

        
        sa:SimulatedAnnealing = SimulatedAnnealing(
            solucion_inicial= random.uniform(-10,10),
            temperatura         = 100.0,
            temperatura_minima  = .0001,
            factor_enfriamiento = 0.95,
            axo_key             = axo_key,
            axo_bucket_id       = axo_bucket_id,
            axo_endpoint_id     = axo_endpoint_id
        )
        
        sa.append_dependency("matplotlib==3.9.2")
        
        
        persistify_result = await sa.persistify()
        assert persistify_result.is_ok
        
        res = sa.simulated()
        assert res.is_ok
        
        res_plot = sa.plot(
            sink_bucket_id  = sink_bucket_id,
            sink_key        = sink_key,
        )
        assert res_plot.is_ok


@pytest.mark.asyncio
async def test_local_simulated_annealing():
    with AxoContextManager.local() as lcm:
        sa:SimulatedAnnealing = SimulatedAnnealing(
            solucion_inicial= random.uniform(-10,10),
            temperatura=100.0,
            temperatura_minima=.0001,
            factor_enfriamiento= 0.95,
            axo_endpoint_id = "axo-endpoint-0"
        )
        res = sa.simulated()
        print(res)
        assert res.is_ok
        solucion, coste, iteraciones  = res.unwrap()
        
        print(f"Solución encontrada: x = {solucion:.5f}")
        print(f"Coste final: f(x) = {coste:.5f}")
        print(f"Iteraciones: {iteraciones}")

        plt.plot(sa.historial_costes) # Aqui hay pedo
        plt.title("Evolución del coste en Simulated Annealing")
        plt.xlabel("Iteración")
        plt.ylabel("Coste f(x)")
        plt.grid(True)
        plt.show()
        
        

@pytest.mark.asyncio
async def test_simulated_annealing_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"

        
        sa:SimulatedAnnealing = SimulatedAnnealing(
            solucion_inicial= random.uniform(-10,10),
            temperatura         = 100.0,
            temperatura_minima  = .0001,
            factor_enfriamiento = 0.95,
            axo_key             = axo_key,
            axo_bucket_id       = axo_bucket_id,
            axo_endpoint_id     = axo_endpoint_id
        )
        
        sa.append_dependency("matplotlib==3.9.2")
        
        
        persistify_result = await sa.persistify()
        assert persistify_result.is_ok
        
        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_res.is_ok
        sa = ao_res.unwrap()
        
        res = sa.simulated()
        assert res.is_ok
        
        res_plot = sa.plot(
            sink_bucket_id  = sink_bucket_id,
            sink_key        = sink_key,
        )
        assert res_plot.is_ok
