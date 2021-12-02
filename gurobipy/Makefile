all: | instances gurobi.lic

.PHONY: run
run: | instances gurobi.lic
	GRB_LICENSE_FILE=./gurobi.lic \
	python test.py

gurobi.lic:
	$(error "Licensa gurobi deve ser inserida na raiz do projeto como gurobi.lic. Para obter uma licensa, visite: https://www.gurobi.com/academia/academic-program-and-licenses/"")
	
instances: instancias-parte2.zip
	unzip instancias-parte2.zip -d instances

install:
	pip install -r requirements.txt

zip:
	rm -f trab2-gurobipy.zip
	zip trab2-gurobipy.zip instancias-parte2.zip requirements.txt test.py Makefile .gitignore

.PHONY: clean
clean:
	rm -r instances .mypy_cache .pycache __pycache__