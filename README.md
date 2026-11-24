## DSTS Final Project - 3312605
Python version used: 3.11.17

Overview: Part A contains notebook code for ML part of the project, which predicts
rating on a given dataset. Part B build a RAG system on a chosen document (ACT Road 
Rules 2017). The document is taken from the ACT government website in DOCX format 
(see SOURCES.md in Part B). They are converted to txt with [online tool](https://www.freeconvert.com/word-to-txt) and split to
3 documents (7 parts each).


## Part A
- src/part_a.ipynb: Initial code, pipeline and plotting
- src/train.py: Classification training pipeline, run CV pipelines and save the best models
- src/predict.py: Classficaition prediction pipeline, load models and predict without training
- src/allocate_data.py: Given allocation code
- data/: raw and allocated datasets and their info
- models/: saved models and preprocesors
- outputs/: saved interactive plot and csv files
- Dockerfile and .dockerignore: Docker build files


All commands to be executed from part_a folder

- install required packages: `pip install -r requirements.txt`
(requirements_docker.txt is only for docker environment, which requires less packages)
- run train.py locally: `python src/train.py`
- run predict.py locally: `python src/predict.py`

Build part_a docker image, using the supplied Dockerfile:

- docker build -t part_a .

Run docker, train and test pipelines: Both use bind mounts with the host's folders 
- `docker run --rm -v "./data/raw:/app/data/raw:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:rw" part_a python src/train.py`

- `docker run --rm -v "./data/allocated:/app/data/allocated:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:ro" part_a python src/predict.py`


## Part B
- notebooks/code.ipynb: RAG pipeline for answering the project questions
- src/app.py: Code for running the TA Road Rules QA app.
- src/road_rules_ta.py: Main logic of the QA app.
- assets/: ACT logo for the QA app
- data/: ACT Road Rules data for the project
- gold/: gold questions and sections
- index/: Chroma indices
- outputs/: csv and json outputs of the notebook and app
- replay/: replay jsons for Replay mode of the app
- Dockerfile and .dockerignore: Docker build files 


All commands to be executed from part_b folder

- install required packages: `pip install -r requirements.txt`
- run the QA app locally: `python -m streamlit run src/app.py`

Build part_a docker image, using the supplied Dockerfile:
- `docker build -t part_b .`

Run the QA app with port 8501 exposed:
- `docker run -p 8501:8501 --env-file .env part_b`
