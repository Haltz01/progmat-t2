all: | instancias-parte2 gurobi.lic gurobi

.PHONY: run
run: | instancias-parte2 gurobi.lic gurobi
	GUROBI_HOME=./gurobi/linux64 \
	PATH="$(GUROBI_HOME)/bin:$PATH" \
	LD_LIBRARY_PATH="$(GUROBI_HOME)/lib:$(LD_LIBRARY_PATH)" \
	GRB_LICENSE_FILE=./gurobi.lic \
	python gurobi_entrega2.py

gurobi.lic:
	$(error "Licensa gurobi deve ser inserida na raiz do projeto como gurobi.lic. Para obter uma licensa, visite: https://www.gurobi.com/academia/academic-program-and-licenses/"")
	
instancias-parte2: instancias-parte2.zip
	unzip instancias-parte2.zip -d instancias-parte2

gurobi: | gurobi9.5.0_linux64.tar.gz
	@echo "Instalando gurobi para linux64..."
	tar -xzf gurobi9.5.0_linux64.tar.gz
	mv gurobi950 gurobi

install:
	pip install -r requirements.txt

zip:
	rm -f trab2.zip
	zip trab2.zip gurobi9.5.0_linux64.tar.gz instancias-parte2.zip requirements.txt gurobi_entrega2.py Makefile .gitignore

sourcezip:
	rm -f trab2-src.zip
	zip trab2-src.zip instancias-parte2.zip requirements.txt gurobi_entrega2.py Makefile .gitignore