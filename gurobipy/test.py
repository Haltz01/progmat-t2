from collections import namedtuple
import dataclasses
from difflib import restore
from enum import Enum, IntEnum
from typing import Dict, List
import gurobipy as gp
from dataclasses import dataclass
import datetime

### CONSTANTES ###   
INSTANCE_NAMES = ["d60900", "d201600", "d401600", "d801600", "e60900", "e801600"]

MAX_SECONDS = 3 * 60  # 3min convertido para segundos -> tempo máximo para cada instância
MAX_AGENTS = 80  # número máximo de agentes

TIMES_TO_RUN_EACH_PRESET = 2

### CLASSES E TIPOS ###

VarDict = Dict[int, Dict[int, gp.Var]]
Run = namedtuple('Run', ['name', 'params', 'output_file'])


@dataclass
class Instance:
    name: str
    nb_agents: int
    nb_tasks: int
    profits: List[List[int]]
    capacityReductions: List[List[int]]
    totalCaps: List[int]

# case_name, exec_time, sol_text, best_result, nb_explored_nodes, best_expected, gap
@dataclass
class InstanceResult:
    instance: Instance
    exec_time: float
    status: int
    best_result: float
    nb_explored_nodes: int
    best_expected: float
    gap: float 

class Method(IntEnum):
    Auto = gp.GRB.METHOD_AUTO
    PrimalSimplex = gp.GRB.METHOD_PRIMAL
    DualSimplex = gp.GRB.METHOD_DUAL
    Barrier = gp.GRB.METHOD_BARRIER
    Concurrent = gp.GRB.METHOD_CONCURRENT

class Cuts(IntEnum):
    Auto = -1 # -1 = automatic setting (solver defines level of aggressiveness in the cut generation)
    NoCuts = 0 # 0 = no cuts
    Moderate = 1 # 1 = moderate cut generation
    Aggressive = 2 # 2 = aggressive cut generation
    VeryAggressive = 3 # 3 = very aggressive cut generation

#Determines which child node is explored first in the branch-and-cut search. The default value chooses automatically. A value of -1 will always explore the down branch first, 
#while a value of 1 will always explore the up branch first.
class BranchDir(IntEnum):  
    Auto = 0
    DownFirst = -1 # -1 always explores down branch first
    UpFirst = 1 # 1 always explorers up branch first

    
# VarBranch controls the branch variable selection strategy
class VarBranch(IntEnum):
    Auto = -1 # -1  = automatic (depending on problem characteristics)
    ReducedCost = 0 # 0 = Pseudo Reduced Cost Branching
    PseudoShadowPrice = 1 # 1 = Pseudo Shadow Price Branching
    MaximumInfeasibility = 2 # 2 = Maximum Infeasibility Branching
    Strong = 3 # 3 = Strong Branching

@dataclass
class SolverParams:
    presolve: bool
    method: Method
    cuts: Cuts
    var_branch: VarBranch
    branch_dir: BranchDir

### FUNÇÕES ###

def read_instance(filename):
    with open(filename, 'r') as input_reader:
        first_line = input_reader.readline()
        content_lines = input_reader.readlines()

        agents_and_tasks = first_line.split(" ")
        nb_agents = int(agents_and_tasks[0])
        nb_tasks = int(agents_and_tasks[1])

        profits = []
        capacityReductions = []
        totalCaps = []

        for i in range(nb_agents):
            profit_line = content_lines[i]
            capRed_line = content_lines[i + nb_agents]

            agent_profits = [int(profit)
                                 for profit in profit_line.split(" ")[:-1]]
            agent_capReds = [int(capRed)
                                 for capRed in capRed_line.split(" ")[:-1]]

            profits.append(agent_profits)
            capacityReductions.append(agent_capReds)

        total_cap_vec = content_lines[-1].split(" ")
        for i in range(nb_agents):
            totalCaps.append(int(total_cap_vec[i]))

        return Instance(filename, nb_agents, nb_tasks, profits, capacityReductions, totalCaps)




def insert_x_variables(model: gp.Model, instance: Instance) -> VarDict:
    x_vars: VarDict = dict()
    for i in range(instance.nb_agents):
        x_vars[i] = dict()
        for j in range(instance.nb_tasks):
            x_vars[i][j] = model.addVar(vtype=gp.GRB.BINARY,
                                      name="x_{}_{}".format(i, j))
    return x_vars


def insert_restrictions(model: gp.Model, instance: Instance, x_vars: VarDict) -> None:
    # Σ_{i=1}^m x_{ij} = 1, j = 1, 2, ..., n -> cada tarefa j só é executada por um agente
    for j in range(instance.nb_tasks):
        model.addConstr(gp.quicksum(x_vars[i][j]
                        for i in range(instance.nb_agents)) == 1)

    # Σ_{j=1}^n a_{ij} * x_{ij} <= cap_{i}, i = 1, 2, ..., m -> cada agente i não pode executar mais tarefas do que a sua capacidade
    for i in range(instance.nb_agents):
        agent_capReds = instance.capacityReductions[i]
        model.addConstr(gp.quicksum(x_vars[i][j] * agent_capReds[j]
                        for j in range(instance.nb_tasks)) <= instance.totalCaps[i])

    # x_{ij} E {0,1}, i = 1, 2, ..., n; j = 1, 2, ..., n -> restrição garantida pelo var_type=BINARY ao criar a variável de decisão
    ###


def insert_objective(model: gp.Model, instance: Instance, x: VarDict):
    model.setObjective(

        gp.quicksum(
            instance.profits[i][j] * x[i][j]
                for i in range(instance.nb_agents)
                    for j in range(instance.nb_tasks)
        ),
        gp.GRB.MAXIMIZE
    )


def setup_instance_model(instance: Instance, solver_params: SolverParams) -> gp.Model:
    model = gp.Model()

    x_vars = insert_x_variables(model, instance)
    insert_restrictions(model, instance, x_vars)
    insert_objective(model, instance, x_vars)
    # model.setParam('OutputFlag', False)

    model.setParam(gp.GRB.param.Presolve, 1 if solver_params.presolve else 0)
    model.setParam(gp.GRB.param.Method, solver_params.method)
    model.setParam(gp.GRB.param.TimeLimit, MAX_SECONDS)
    model.setParam(gp.GRB.param.Threads, 1)
    model.setParam(gp.GRB.param.Cuts, solver_params.cuts)
    model.setParam(gp.GRB.param.BranchDir, solver_params.branch_dir)
    return model

def try_pass(lmbd):
    try:
        lmbd()
    except Exception as e:
        print("[ERROR] try_print Exception: {}".format(e))
        pass


def solve_instance(instance: Instance, model: gp.Model) -> InstanceResult:
    model.optimize()

    '''
    # Principais resultados
    print("\tValor da função objetivo: {}".format(model.objVal))
    try_pass(lambda: print(f'gp.GRB.Attr.MIPGap: {model.getAttr(gp.GRB.Attr.MIPGap)}')) # GAP
    try_pass(lambda: print(f'gp.GRB.Attr.MaxBound: {model.getAttr(gp.GRB.Attr.MaxBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NodeCount: {model.getAttr(gp.GRB.Attr.NodeCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinBound: {model.getAttr(gp.GRB.Attr.MinBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinCoeff: {model.getAttr(gp.GRB.Attr.MinCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Runtime: {model.getAttr(gp.GRB.Attr.Runtime)}')) # Exec time
    # try_pass(lambda: print(f'gp.GRB.Attr.Obj: {model.getAttr(gp.GRB.Attr.Obj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjBound: {model.getAttr(gp.GRB.Attr.ObjBound)}')) # limitante dual
    try_pass(lambda: print(f'gp.GRB.Attr.ObjVal: {model.getAttr(gp.GRB.Attr.ObjVal)}')) # melhor resultado até agora
    try_pass(lambda: print(f'gp.GRB.Attr.IterCount: {model.getAttr(gp.GRB.Attr.IterCount)}')) # número de iterações do simplex (nós explorados)
    # try_pass(lambda: print(f'gp.GRB.Attr.BranchPriority: {model.getAttr(gp.GRB.Attr.BranchPriority)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Status: {model.getAttr(gp.GRB.Attr.Status)}'))

    # Resultados secundários
    # try_pass(lambda: print(f'gp.GRB.Attr.IsMIP: {model.getAttr(gp.GRB.Attr.IsMIP)}'))
    '''

    instance_result = InstanceResult(
        instance=instance,
        exec_time=model.getAttr(gp.GRB.Attr.Runtime),
        status=model.getAttr(gp.GRB.Attr.Status),
        best_result=model.getAttr(gp.GRB.Attr.ObjVal),
        nb_explored_nodes=model.getAttr(gp.GRB.Attr.NodeCount),
        best_expected=model.getAttr(gp.GRB.Attr.ObjBound),
        gap=model.getAttr(gp.GRB.Attr.MIPGap)
    )
    return instance_result

def status_to_text(status: int):
    if status == gp.GRB.Status.CUTOFF: return "cutoff"
    elif status == gp.GRB.Status.INFEASIBLE: return "infeasible solution"
    elif status == gp.GRB.Status.INF_OR_UNBD: return "inf. or unbd."
    elif status == gp.GRB.Status.INPROGRESS: return "inprogress"
    elif status == gp.GRB.Status.INTERRUPTED: return "interrupted execution"
    elif status == gp.GRB.Status.ITERATION_LIMIT: return "iteration limit reached"
    elif status == gp.GRB.Status.LOADED: return "loaded"
    elif status == gp.GRB.Status.NODE_LIMIT: return "node limit reached"
    elif status == gp.GRB.Status.NUMERIC: return "numeric"
    elif status == gp.GRB.Status.OPTIMAL: return "optimal solution"
    elif status == gp.GRB.Status.SOLUTION_LIMIT: return "solution qty. limit reached"
    elif status == gp.GRB.Status.SUBOPTIMAL: return "suboptimal solution"
    elif status == gp.GRB.Status.TIME_LIMIT: return "time limit reached"
    elif status == gp.GRB.Status.UNBOUNDED: return "unbounded result"
    elif status == gp.GRB.Status.USER_OBJ_LIMIT: return "user obj. limit reached"
    elif status == gp.GRB.Status.WORK_LIMIT: return "work limit reached"
    else: raise ValueError(f'Unknown status: {status}')

def test_first_instance():
    instance = read_instance(f'instances/{INSTANCE_NAMES[0]}.in')

    print("\n\tInstância: {}\n".format(INSTANCE_NAMES[0]))

    print(f'\tNúmero de agentes: {instance.nb_agents}')
    print(f'\tNúmero de tarefas: {instance.nb_tasks}')
    print(f'\tÚltimo lucro: {instance.profits[-1][-1]}')
    print(f'\tÚltima redCap: {instance.capacityReductions[-1][-1]}')
    print(f'\tÚltima totalCap: {instance.totalCaps[-1]}')

# model = setup_instance_model(instance)
# solve_instance(instance, model)

def write_instance_result(result: InstanceResult, output_file_name: str):
    solution_text = status_to_text(result.status)
    curr_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #           data_atual,        caso_teste,          tempo_total,      conclusao,     melhor_resultado,       num_nos_explorados,       limitante_dual,       gap
    line = f'{curr_date_str},{result.instance.name},{result.exec_time:.3f},{solution_text},{result.best_result:.3f},{result.nb_explored_nodes:.0f},{result.best_expected:.3f},{result.gap * 100:.3f}'

    with open(output_file_name, 'a') as output_file:
        output_file.write(line + '\n')
        output_file.flush()
    
    return

def init_results_file(preset_name: str, output_file_name):
    zero_line = "Preset usado: " + preset_name
    first_line = 'data_atual,caso_teste,tempo_total,conclusao,melhor_resultado,num_nos_explorados,limitante_dual,gap(%)'

    with open(output_file_name, 'a') as output_file:
        output_file.write(zero_line + '\n')
        output_file.write(first_line + '\n')

    return

def run_all_instances_with_params(run: Run):
    init_results_file(run.name, run.output_file)

    solver_params = run.params
    for instance_name in INSTANCE_NAMES:
        print("\n\tInstância: {}\n".format(instance_name))

        instance = read_instance(f"instances/{instance_name}.in")
        model = setup_instance_model(instance, solver_params)
        result = solve_instance(instance, model)
        write_instance_result(result, run.output_file)

# Seleção de método utilizado pelo Solver -> https://www.gurobi.com/documentation/9.1/refman/method.html#parameter:Method
# Seleção de parâmetros -> https://www.gurobi.com/documentation/8.0/refman/parameters.html#sec:Parameters 
def main():
    default_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.Auto
    )
    
    no_preprocess_params = SolverParams(
        presolve=False,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.Auto
    )
    
    no_cuts_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.NoCuts,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.Auto
    )

    varbrach_redcost_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.ReducedCost,
        branch_dir=BranchDir.Auto
    )

    varbrach_strong_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.Strong,
        branch_dir=BranchDir.Auto
    )

    dir_up_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.UpFirst
    )

    dir_down_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Auto,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.DownFirst
    )

    cuts_agressive_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.Aggressive,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.Auto
    )

    cuts_very_agressive_params = SolverParams(
        presolve=True,
        method=Method.Auto,
        cuts=Cuts.VeryAggressive,
        var_branch=VarBranch.Auto,
        branch_dir=BranchDir.Auto
    )

    runs = [
        Run('Default', default_params, 'results/default.csv'),
        Run('No Preprocess', no_preprocess_params, 'results/no_preprocess.csv'),
        Run('No Cuts', no_cuts_params, 'results/no_cuts.csv'),
        Run('VarBranch Reducted Cost', varbrach_redcost_params, 'results/varbranch_redcost.csv'),
        Run('VarBranch Strong', varbrach_strong_params, 'results/varbranch_strong.csv'),
        Run('Branch Up', dir_up_params, 'results/branch_up.csv'),
        Run('Branch Down', dir_down_params, 'results/branch_down.csv'),
        Run('Cuts Agressive', cuts_agressive_params, 'results/cuts_agressive.csv'),
        Run('Cuts Very Agressive', cuts_very_agressive_params, 'results/cuts_very_agressive.csv')
    ]

    for run in runs:
        print(f'\nExecutando solver com predefinições: {run.name}')
        for _ in range(TIMES_TO_RUN_EACH_PRESET):
            run_all_instances_with_params(run)


if __name__ == "__main__":
    main()
    # print(f'{gp.GRB.METHOD_AUTO=}')
    # print(f'{gp.GRB.METHOD_BARRIER=}')
    # print(f'{gp.GRB.METHOD_CONCURRENT=}')
    # print(f'{gp.GRB.METHOD_DUAL=}')
    # print(f'{gp.GRB.METHOD_DETERMINISTIC_CONCURRENT=}')
    # print(f'{gp.GRB.METHOD_NONE=}')
    # print(f'{gp.GRB.METHOD_PRIMAL=}')