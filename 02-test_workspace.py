from azureml.core import Workspace, Experiment, ScriptRunConfig, experiment

ws = Workspace.from_config(path='./.azureml', _file_name='config.json')

experiment = Experiment(workspace=ws, name='day1-experiment-testing-homework')

config = ScriptRunConfig(source_directory='./src', script='test_ws.py', compute_target='test-cluster')

run = experiment.submit(config)

aml_url = run.get_portal_url()

print(aml_url)