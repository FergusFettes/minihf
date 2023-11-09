all: venv install

venv:
	apt-get update
	-apt-get upgrade -y
	-apt-get install python3.10-venv -y
	python3 -m venv venv

install:
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && flask --app minihf_infer run
