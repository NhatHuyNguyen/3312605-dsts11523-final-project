Python version used: 3.11.17

notebooks/code.ipynb: RAG pipeline for answering the project questions
src/app.py: Code for running the TA Road Rules QA app.
src/road_rules_ta.py: Main logic of the QA app.
assets/: ACT logo for the QA app
data/: ACT Road Rules data for the project
gold/: gold questions and sections
index/: Chroma indices
outputs/: csv and json outputs of the notebook and app
replay/: replay jsons for Replay mode of the app
Dockerfile and .dockerignore: Docker build files 


All commands to be executed from part_b folder

- Install required packages: pip install -r requirements.txt
- Run the QA app natively: python -m streamlit run src/app.py

Command to build part_a docker image, using the supplied Dockerfile:
- docker build -t part_b .

Commands to run the QA app in Docker with port 8501 exposed:
- docker run -p 8501:8501 --env-file .env part_b
