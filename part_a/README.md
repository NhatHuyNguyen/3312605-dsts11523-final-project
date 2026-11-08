Python version used: 3.11.17

src/part_a.ipynb: Initial code, pipeline and plotting
src/train.py: Classification training pipeline, run CV pipelines and save the best models
src/predict.py: Classficaition prediction pipeline, load models and predict without training
src/allocate_data.py: Given allocation code
data/: raw and allocated datasets and their info
models/: saved models and preprocesors
outputs/: saved interactive plot and csv files
Dockerfile and .dockerignore: Docker build files


All commands to be executed from part_a folder

- Install required packages: pip install -r requirements.txt
(requirements_docker.txt is only for docker environment, which requires less packakges)
- Run train.py: python src/train.py
- Run predict.py: python src/predict.py

Command to build part_a docker image, using the supplied Dockerfile:
- docker build -t part_a .

Commands to run docker, train and test pipelines: Both use bind mounts with the host's folders 
- docker run --rm -v "./data/raw:/app/data/raw:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:rw" part_a python src/train.py

- docker run --rm -v "./data/allocated:/app/data/allocated:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:ro" part_a python src/predict.py
