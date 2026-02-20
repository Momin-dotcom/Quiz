setup:
	pip install -r requirements.txt

preprocess:
	python src/preprocess.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

all: preprocess train evaluate