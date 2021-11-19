from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id='99e1e721-7184-498e-8aff-b2ad4e53c1c2')
ws = Workspace.get(
    name='mlw-esp-udea-tarea',
    subscription_id='34772c50-7adb-4ef8-a18f-8044390834c4',
    resource_group='rg-ml-esp-udea-tarea',
    location='eastus2',
    auth=interactive_auth
)
ws.write_config(path='.azureml')


