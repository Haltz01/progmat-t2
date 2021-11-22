all: instancias-parte2

run: instancias-parte2
	python gurobi_entrega2.py

instancias-parte2: instancias-parte2.zip
	unzip instancias-parte2.zip -d instancias-parte2