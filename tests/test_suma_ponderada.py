from axo.contextmanager import AxoContextManager
from optikit.algorithms.suma_ponderada import SumaPonderada 
import pytest
import numpy as np
import pytest_asyncio
from axo.endpoint.manager import DistributedEndpointManager
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
async def test_suma_ponderada():
    with AxoContextManager.local() as dcm:
        sp:SumaPonderada = SumaPonderada(
        
            axo_endpoint_id="axo-endpoint-0")
        _ = await sp.persistify()
        res = sp.suma(pasos=50)
        print("Frente de Pareto generado con éxito:", res)
        assert res.is_ok
        solutions = res.unwrap()
        
        
@pytest.mark.asyncio
async def test_distributed_suma_ponderada(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao1_sink_bucket"
        sink_key                = "ao1_sink_key"
        axo_key                 = "ao1"
        axo_bucket_id           = "ao1bucket"
        
        sp:SumaPonderada = SumaPonderada(
            axo_key             = axo_key,
            axo_bucket_id       = axo_bucket_id,
            axo_endpoint_id     = axo_endpoint_id
        )
        
        sp.append_dependency("matplotlib==3.9.2")
        sp.append_dependency("numpy==1.24.3")
    
        persistify_result = await sp.persistify()
        assert persistify_result.is_ok
    
        res = sp.suma(pasos=50)
        assert res.is_ok
    
        soluciones = res.unwrap()
    
        res_plot = sp.plot(
        soluciones,
        sink_bucket_id  = sink_bucket_id,
        sink_key        = sink_key,
        )
        assert res_plot.is_ok
        

@pytest.mark.asyncio
async def test_distributed_suma_ponderada_zk(endpoint_manager:DistributedEndpointManager,storage_service:MictlanXStorageService):
    with AxoContextManager.distributed(endpoint_manager=endpoint_manager,storage_service=storage_service) as dcm:
        
        axo_endpoint_id         = "axo-endpoint-0"
        sink_bucket_id          = "ao2_sink_bucket"
        sink_key                = "ao2_sink_key"
        axo_key                 = "ao2"
        axo_bucket_id           = "ao2bucket"
        
        sp:SumaPonderada = SumaPonderada(
            axo_key             = axo_key,
            axo_bucket_id       = axo_bucket_id,
            axo_endpoint_id     = axo_endpoint_id
        )
        
        sp.append_dependency("matplotlib==3.9.2")
        sp.append_dependency("numpy==1.24.3")
    
        persistify_result = await sp.persistify()
        assert persistify_result.is_ok
    
        ao_res = await Axo.get_by_key(bucket_id=axo_bucket_id,key=axo_key)
        assert ao_res.is_ok
        
        sp = ao_res.unwrap()
        
        res = sp.suma(pasos=50)
        assert res.is_ok
    
        soluciones = res.unwrap()
    
        res_plot = sp.plot(
        soluciones,
        sink_bucket_id  = sink_bucket_id,
        sink_key        = sink_key,
        )
        assert res_plot.is_ok