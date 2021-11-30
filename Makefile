all: instancias-parte2 gurobi

run: instancias-parte2 gurobi
	GUROBI_HOME=./gurobi/linux64 \
	PATH="$(GUROBI_HOME)/bin:$PATH" \
	LD_LIBRARY_PATH="$(GUROBI_HOME)/lib:$(LD_LIBRARY_PATH)" \
	GRB_LICENSE_FILE=./gurobi.lic \
	python gurobi_entrega2.py

instancias-parte2: instancias-parte2.zip
	unzip instancias-parte2.zip -d instancias-parte2

gurobi: gurobi9.5.0_linux64.tar.gz
	tar -xzf gurobi9.5.0_linux64.tar.gz
	mv gurobi950 gurobi

install:
	pip install -r requirements.txt

zip:
	rm -f trab2.zip
	zip trab2.zip gurobi9.5.0_linux64.tar.gz instancias-parte2.zip requirements.txt gurobi_entrega2.py Makefile .gitignore