all: venv install run

venv:
	apt-get update -y
	apt-get upgrade -y
	apt-get install python3.10-venv -y
	python3 -m venv venv

install:
	source venv/bin/activate && pip install -r requirements.txt

run:
	source venv/bin/activate && flask --app minihf_infer run
