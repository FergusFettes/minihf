all: venv hosting install

venv:
	apt-get update
	-apt-get upgrade -y
	-apt-get install python3.10-venv -y
	python3 -m venv venv

hosting:
	-apt-get install -y software-properties-common
	add-apt-repository ppa:jgmath2000/et -y
	apt-get update
	-apt-get install -y mosh et
	snap install ngrok

electron:
	-apt-get install -y npm
	npm install -g electron

install:
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && flask --app minihf_infer run

run_fast:
	. venv/bin/activate && uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

host:
	ngrok config add-authtoken $(TOKEN)
	ngrok http 5000
