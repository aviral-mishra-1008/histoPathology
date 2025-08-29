venv:
	python -m venv histoVenv

install:
	. .\histoVenv\Scripts\Activate.ps1; pip install -r requirements.txt

run:
	. .\histoVenv\Scripts\Activate.ps1; python run.py

all: venv install run
