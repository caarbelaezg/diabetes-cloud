# Instructions

When downloadning create 'outputs' folder in root. After that run diabetes_train.py from src folder

## Runing local with docker
1. Copy or move the model from outputs folder to docker folder, do not change the name.
2. Execute using command prompt inside docker folder 
    ```
    docker build -t diabetes-model . 
    docker run -p 3000:3000 diabetes-model  
    ```