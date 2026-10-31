Python version used: 3.11.17

src/part_a.ipynb: Initial code, pipeline and plotting
src/train.py: Classification training pipeline, run CV pipelines and save the best models
src/predict.py: Classficaition prediction pipeline, load models and predict without training

Command to install required packages, run from part_a: pip install -r requirements.txt
Command to run train.py, run from part_a: python src/train.py
Command to run predict.py, run from part_a: python src/train.py

Commands to run docker, train and test pipelines: 
- docker run --rm -v "./data/raw:/app/data/raw:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:rw" part_a python src/train.py

- docker run --rm -v "./data/allocated:/app/data/allocated:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:ro" part_a python src/predict.py
