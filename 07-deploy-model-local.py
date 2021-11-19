from azureml.core import environment
from azureml.core.webservice import webservice
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core import Workspace, workspace
from azureml.core.model import Model
from azureml.core.webservice import LocalWebservice

ws = Workspace.from_config(path='./.azureml', _file_name='config.json')
model = Model(ws, name='diabetes_model', version=1)

env = Environment.from_conda_specification(
    name='diabetes-remote-env',
    file_path='./.azureml/diabetes-remote.yml'
)

inference_config = InferenceConfig(entry_script='./src/score.py', environment=env)

deployment_config = LocalWebservice.deploy_configuration(port=3000)

local_service = Model.deploy(workspace=ws,
                        name='diabetes-model-local',
                        models=[model],
                        inference_config=inference_config,
                        deployment_config=deployment_config
                        )

local_service.wait_for_deployment(show_output=True)
print(f'Scoring URI is: {local_service.scoring_uri}')