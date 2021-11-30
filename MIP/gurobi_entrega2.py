#!/usr/bin/python3

# property "preprocess"
# Enables/disables pre-processing. Pre-processing tries to improve your MIP formulation. -1 means automatic, 0 means off and 1 means on.

from mip import * # https://docs.python-mip.com/en/latest/index.html
import time
import re

cases_name = ["d60900", "d201600", "d401600", "d801600", "e60900", "e801600" ]

MAX_SECONDS = 3 * 60 # 3min convertido para segundos -> tempo máximo para cada instância
MAX_AGENTS = 80 # número máximo de agentes

def solveInstance(filename):
    print("[!] Iniciando resolução com arquivo de entrada '{}'".format(filename))
    input_reader = open(filename, "r")

    first_line = input_reader.readline()
    content_lines = input_reader.readlines()

    input_reader.close() # fechando arquivo de entrada

    agents_and_tasks = first_line.split(" ")
    nb_agents = int(agents_and_tasks[0])
    nb_tasks = int(agents_and_tasks[1])
    print("\tNúmero de agentes = {}\n\tNúmero de tarefas = {}".format(nb_agents, nb_tasks))

    # para cada agente i (i=1,...,m):
    # satisfacao (c_{ij} = lucro) por alocar a tarefa j ao agente i (j=1,...,n)
    profits_agents = [] # cada linha desse array 2D representa o lucro gerado por um agente ao realizar cada tarefa
    for i in range(nb_agents):
        profits_agents.append(list(map(int, content_lines[i].split(" ")[:-1])))
    
    # para cada agente i (i=1,...,m):
    # recurso consumido (a_{ij}) ao alocar a tarefa j ao agente i (j=1,...,n) 
    effort_agents = [] # cada linha desse array 2D representa o recurso consumido por um agente ao realizar cada tarefa
    for i in range(nb_agents):
        effort_agents.append(list(map(int, content_lines[nb_agents + i].split(" ")[:-1])))
    
    # capacidade do agente i (i=1,...,m)
    capacities_agents = [] # cada item (de índice i) desse array representa a capacidade de realização de tarefas do agente i
    capacities_array_line_index = nb_agents * 2
    capacities_array = content_lines[capacities_array_line_index].split(" ")
    for i in range(nb_agents):
        capacities_agents.append(int(capacities_array[i]))
    
    for i in range(nb_agents):
        print("\tAgente {} com capacidade = {}".format(i, capacities_agents[i]))

    # ===== Criação do modelo =====
    # sense=MAXIMIZE -> queremos maximizar nossa função objetiva
    # GRB = Gurobi
    # CBC = Coin-Or Branch and Cut
    # export GUROBI_HOME="/home/haltz/Downloads/USP - 2º semestre 2021/ProgMat/gurobi_lib/gurobi9.1.2_linux64/gurobi912/linux64/"
    # export GRB_LICENSE_FILE="/home/haltz/Downloads/USP - 2º semestre 2021/ProgMat/gurobi_lib/gurobi.lic" -> usar a licença do Gurobi
    model = Model(sense = MAXIMIZE, solver_name = GRB)

    model.preprocess = 0 # desabilizando preprocessamento

    # ===== Variáveis de decisão =====
    decision_variables = []

    for i in range(nb_agents): # criando os x_{ij} no qual x = agente executa ou não tarefa (1 ou 0), i = agente e j = tarefa
        decision_variables.append([ model.add_var(name='X[{}][{}]'.format(i, j), var_type=BINARY) for j in range(nb_tasks) ]) # para cada agente, cria uma lista de variáveis de decisão de tamanho "nb_tasks" (= qtd. de tarefas disponíveis) 
    
    # ===== Restrições =====
        
    # Σ_{i=1}^m x_{ij} = 1, j = 1, 2, ..., n -> cada tarefa j só é executada por um agente
    for j in range(nb_tasks):
        model += xsum(decision_variables[i][j] for i in range(nb_agents)) == 1
    
    # Σ_{j=1}^n a_{ij} * x_{ij} <= cap_{i}, i = 1, 2, ..., m -> cada agente i não pode executar mais tarefas do que a sua capacidade
    for i in range(nb_agents):
        model += xsum(effort_agents[i][j] * decision_variables[i][j] for j in range(nb_tasks)) <= capacities_agents[i]
    
    # x_{ij} E {0,1}, i = 1, 2, ..., n; j = 1, 2, ..., n -> restrição garantida pelo var_type=BINARY ao criar a variável de decisão

    # ===== Função Objetivo =====
    # Σ_{i=1}^m Σ_{j=1}^n ( c_{ij} * x_{ij} )
    model.objective = xsum(profits_agents[i][j] * decision_variables[i][j] for j in range(nb_tasks) for i in range(nb_agents))

    # Realizando otimização com tempo limite de MAX_SECONDS
    status = model.optimize(max_seconds=MAX_SECONDS)

    result_text = analyzeResult(status, model)

    return (result_text, model)

def analyzeResult(solve_status, model):
    solution_text = ''
    if solve_status == OptimizationStatus.OPTIMAL: # resultado ótimo encontrado
        solution_text = 'Solução ótima encontrada com "lucro" de {}'.format(model.objective_value)
    elif solve_status == OptimizationStatus.FEASIBLE: # resultado viável encontrado
        solution_text = 'Solução viável encontrada com "lucro" de {} - esperança (melhor possível): {}'.format(model.objective_value, model.objective_bound)
    elif solve_status == OptimizationStatus.NO_SOLUTION_FOUND: # nenhum resultado encontrado
        solution_text = 'Solução factível não encontrada - limite inferior: {}'.format(model.objective_bound)
    
    print('\t' + solution_text)

    #if solve_status == OptimizationStatus.OPTIMAL or solve_status == OptimizationStatus.FEASIBLE:
    #    print('\n=== Solução ===\nObs.: X[i][j] = 1 representa que agente i realiza tarefa j\n')
    #    for var in model.vars:
    #        if abs(var.x) > 1e-6: # só printamos valores diferentes de zero
    #            print('{} : {}'.format(var.name, var.x))
    
    return solution_text

# Salvando resultados de cada caso de teste
def formatTestcaseCSV(case_name: str, exec_time, sol_text: str, vars):
    #        'caso_teste,tempo_total,conclusao'
    line = f'{case_name},{exec_time},{sol_text}'

    # for var in vars: # variáveis do modelo
    #     if abs(var.x) > 1e-6:
    #         # X[i][j] -> var.name
    #         indices = re.findall(r'\[(\d+)]\[(\d+)]', var.name)[0]
    #         assert len(indices) >= 2, 'var.name malformed! should be X[i][j]'
    #         i, j = indices
    #         line += f',{j}'
    
    return line

# Preparando primeira linha do arquivo de saída CSV
first_line = 'caso_teste,tempo_total,conclusao'
# for i in range(MAX_AGENTS):
#     first_line += ',agente{}'.format(i)

output_file = open('gurobi_entrega2.csv', 'w+')
output_file.write(first_line + '\n')

for case_name in cases_name:
    start_time = time.time()
    
    result_text, model = solveInstance("instancias-parte2/" + case_name + ".in")

    exec_time = time.time() - start_time # tempo gasto resolvendo o caso de teste

    output_file.write(formatTestcaseCSV(case_name, exec_time, result_text, model.vars) + '\n')
    output_file.flush()

print("Finalizando programa...")
output_file.close() 