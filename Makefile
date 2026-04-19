install:
	pip install -r requirements.txt

data:
	python generate_data.py

train:
	python main.py

api:
	uvicorn api.app:app --reload

test:
	pytest -v

smoke:
	python scripts/smoke_test.py