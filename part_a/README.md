

Commands to run docker
- docker run --rm -v "./data/raw:/app/data/raw:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:rw" part_a:v1.0 python src/train.py

- docker run --rm -v "./data/allocated:/app/data/allocated:ro" -v "./outputs:/app/outputs:rw" -v "./models:/app/models:ro" part_a:v1.0 python src/predict.py
