import dataclasses
from typing import Dict, List
import gurobipy as gp
from dataclasses import dataclass
import datetime

### CONSTANTES ###   
INSTANCE_NAMES = ["d60900", "d201600", "d401600", "d801600", "e60900", "e801600"]

MAX_SECONDS = 3 * 60  # 3min convertido para segundos -> tempo máximo para cada instância
MAX_AGENTS = 80  # número máximo de agentes

RESULTS_FILENAME = 'results.csv'

### CLASSES E TIPOS ###

VarDict = Dict[int, Dict[int, gp.Var]]

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


def create_standard_model():
    model = gp.Model()
    # model.setParam('OutputFlag', False)
    return model


def create_primal_simplex_model():
    model = create_standard_model()
    model.setParam('Method', 0)
    return model


def create_dual_simplex_model():
    model = create_standard_model()
    model.setParam('Method', 1)
    return model


def create_barrier_model():
    model = create_standard_model()
    model.setParam('Method', 2)
    return model


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


def setup_instance_model(instance: Instance) -> gp.Model:
    model = create_barrier_model()
    x_vars = insert_x_variables(model, instance)
    insert_restrictions(model, instance, x_vars)
    insert_objective(model, instance, x_vars)
    # model.setParam('OutputFlag', False)
    model.setParam('TimeLimit', MAX_SECONDS)
    model.setParam('Threads', 6)
    return model

def try_pass(lmbd):
    try:
        lmbd()
    except Exception as e:
        print("[ERROR] try_print Exception: {}".format(e))
        pass


def solve_instance(instance: Instance, model: gp.Model) -> InstanceResult:
    model.optimize()
    if model.status == gp.GRB.OPTIMAL:
        print("\n\tSolução ótima encontrada!\n")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("\n\tLimite de tempo excedido!\n")
    elif model.status == gp.GRB.INF_OR_UNBD:
        print("\n\tProblema infinito ou não definido!\n")
    elif model.status == gp.GRB.INFEASIBLE:
        print("\n\tProblema inviavel!\n")
    elif model.status == gp.GRB.UNBOUNDED:
        print("\n\tProblema ilimitado!\n")
    else:
        print("\n\tSolução não encontrada! (ERRO DESCONHECIDO)\n")

    print()


    
        

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

    instance_result = InstanceResult(
        instance=instance,
        exec_time=model.getAttr(gp.GRB.Attr.Runtime),
        status=model.status,
        best_result=model.objVal,
        nb_explored_nodes=model.getAttr(gp.GRB.Attr.NodeCount),
        best_expected=model.getAttr(gp.GRB.Attr.ObjBound),
        gap=model.getAttr(gp.GRB.Attr.MIPGap)
    )
    return instance_result

    # Outros resultados
    '''
    try_pass(lambda: print(f'gp.GRB.Attr.BarIterCount: {model.getAttr(gp.GRB.Attr.BarIterCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVio: {model.getAttr(gp.GRB.Attr.BoundSVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVioIndex: {model.getAttr(gp.GRB.Attr.BoundSVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVioSum: {model.getAttr(gp.GRB.Attr.BoundSVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.CTag: {model.getAttr(gp.GRB.Attr.CTag)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConcurrentWinMethod: {model.getAttr(gp.GRB.Attr.ConcurrentWinMethod)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrName: {model.getAttr(gp.GRB.Attr.ConstrName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DNumNZs: {model.getAttr(gp.GRB.Attr.DNumNZs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Fingerprint: {model.getAttr(gp.GRB.Attr.Fingerprint)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceError: {model.getAttr(gp.GRB.Attr.FuncPieceError)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceLength: {model.getAttr(gp.GRB.Attr.FuncPieceLength)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceRatio: {model.getAttr(gp.GRB.Attr.FuncPieceRatio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieces: {model.getAttr(gp.GRB.Attr.FuncPieces)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.GenConstrName: {model.getAttr(gp.GRB.Attr.GenConstrName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.GenConstrType: {model.getAttr(gp.GRB.Attr.GenConstrType)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISConstrForce: {model.getAttr(gp.GRB.Attr.IISConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISGenConstr: {model.getAttr(gp.GRB.Attr.IISGenConstr)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISGenConstrForce: {model.getAttr(gp.GRB.Attr.IISGenConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISLBForce: {model.getAttr(gp.GRB.Attr.IISLBForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISQConstr: {model.getAttr(gp.GRB.Attr.IISQConstr)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISQConstrForce: {model.getAttr(gp.GRB.Attr.IISQConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISSOS: {model.getAttr(gp.GRB.Attr.IISSOS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISSOSForce: {model.getAttr(gp.GRB.Attr.IISSOSForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISUBForce: {model.getAttr(gp.GRB.Attr.IISUBForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVio: {model.getAttr(gp.GRB.Attr.IntVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVioIndex: {model.getAttr(gp.GRB.Attr.IntVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVioSum: {model.getAttr(gp.GRB.Attr.IntVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsMultiObj: {model.getAttr(gp.GRB.Attr.IsMultiObj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsQCP: {model.getAttr(gp.GRB.Attr.IsQCP)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsQP: {model.getAttr(gp.GRB.Attr.IsQP)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.JobID: {model.getAttr(gp.GRB.Attr.JobID)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.LB: {model.getAttr(gp.GRB.Attr.LB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Lazy: {model.getAttr(gp.GRB.Attr.Lazy)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.LicenseExpiration: {model.getAttr(gp.GRB.Attr.LicenseExpiration)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxCoeff: {model.getAttr(gp.GRB.Attr.MaxCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxObjCoeff: {model.getAttr(gp.GRB.Attr.MaxObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCCoeff: {model.getAttr(gp.GRB.Attr.MaxQCCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCLCoeff: {model.getAttr(gp.GRB.Attr.MaxQCLCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCRHS: {model.getAttr(gp.GRB.Attr.MaxQCRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQObjCoeff: {model.getAttr(gp.GRB.Attr.MaxQObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxRHS: {model.getAttr(gp.GRB.Attr.MaxRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxVio: {model.getAttr(gp.GRB.Attr.MaxVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinObjCoeff: {model.getAttr(gp.GRB.Attr.MinObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCCoeff: {model.getAttr(gp.GRB.Attr.MinQCCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCLCoeff: {model.getAttr(gp.GRB.Attr.MinQCLCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCRHS: {model.getAttr(gp.GRB.Attr.MinQCRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQObjCoeff: {model.getAttr(gp.GRB.Attr.MinQObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinRHS: {model.getAttr(gp.GRB.Attr.MinRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ModelName: {model.getAttr(gp.GRB.Attr.ModelName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ModelSense: {model.getAttr(gp.GRB.Attr.ModelSense)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumBinVars: {model.getAttr(gp.GRB.Attr.NumBinVars)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumConstrs: {model.getAttr(gp.GRB.Attr.NumConstrs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumGenConstrs: {model.getAttr(gp.GRB.Attr.NumGenConstrs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumIntVars: {model.getAttr(gp.GRB.Attr.NumIntVars)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumNZs: {model.getAttr(gp.GRB.Attr.NumNZs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumObj: {model.getAttr(gp.GRB.Attr.NumObj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumPWLObjVars: {model.getAttr(gp.GRB.Attr.NumPWLObjVars)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumQCNZs: {model.getAttr(gp.GRB.Attr.NumQCNZs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumQConstrs: {model.getAttr(gp.GRB.Attr.NumQConstrs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumQNZs: {model.getAttr(gp.GRB.Attr.NumQNZs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumSOS: {model.getAttr(gp.GRB.Attr.NumSOS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumScenarios: {model.getAttr(gp.GRB.Attr.NumScenarios)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumStart: {model.getAttr(gp.GRB.Attr.NumStart)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NumVars: {model.getAttr(gp.GRB.Attr.NumVars)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjBoundC: {model.getAttr(gp.GRB.Attr.ObjBoundC)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjCon: {model.getAttr(gp.GRB.Attr.ObjCon)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjN: {model.getAttr(gp.GRB.Attr.ObjN)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNAbsTol: {model.getAttr(gp.GRB.Attr.ObjNAbsTol)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNCon: {model.getAttr(gp.GRB.Attr.ObjNCon)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNName: {model.getAttr(gp.GRB.Attr.ObjNName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNPriority: {model.getAttr(gp.GRB.Attr.ObjNPriority)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNRelTol: {model.getAttr(gp.GRB.Attr.ObjNRelTol)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNVal: {model.getAttr(gp.GRB.Attr.ObjNVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjNWeight: {model.getAttr(gp.GRB.Attr.ObjNWeight)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PStart: {model.getAttr(gp.GRB.Attr.PStart)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PWLObjCvx: {model.getAttr(gp.GRB.Attr.PWLObjCvx)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Partition: {model.getAttr(gp.GRB.Attr.Partition)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Pi: {model.getAttr(gp.GRB.Attr.Pi)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PoolIgnore: {model.getAttr(gp.GRB.Attr.PoolIgnore)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PoolObjBound: {model.getAttr(gp.GRB.Attr.PoolObjBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PoolObjVal: {model.getAttr(gp.GRB.Attr.PoolObjVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.PreFixVal: {model.getAttr(gp.GRB.Attr.PreFixVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCName: {model.getAttr(gp.GRB.Attr.QCName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCPi: {model.getAttr(gp.GRB.Attr.QCPi)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCRHS: {model.getAttr(gp.GRB.Attr.QCRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCSense: {model.getAttr(gp.GRB.Attr.QCSense)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCSlack: {model.getAttr(gp.GRB.Attr.QCSlack)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.QCTag: {model.getAttr(gp.GRB.Attr.QCTag)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.RHS: {model.getAttr(gp.GRB.Attr.RHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Work: {model.getAttr(gp.GRB.Attr.Work)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNLB: {model.getAttr(gp.GRB.Attr.ScenNLB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNName: {model.getAttr(gp.GRB.Attr.ScenNName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNObj: {model.getAttr(gp.GRB.Attr.ScenNObj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNObjBound: {model.getAttr(gp.GRB.Attr.ScenNObjBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNObjVal: {model.getAttr(gp.GRB.Attr.ScenNObjVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNRHS: {model.getAttr(gp.GRB.Attr.ScenNRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNUB: {model.getAttr(gp.GRB.Attr.ScenNUB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ScenNX: {model.getAttr(gp.GRB.Attr.ScenNX)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Sense: {model.getAttr(gp.GRB.Attr.Sense)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Server: {model.getAttr(gp.GRB.Attr.Server)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Slack: {model.getAttr(gp.GRB.Attr.Slack)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SolCount: {model.getAttr(gp.GRB.Attr.SolCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Start: {model.getAttr(gp.GRB.Attr.Start)}'))
    # try_pass(lambda: print(f'gp.GRB.Attr.TuneResultCount: {model.getAttr(gp.GRB.Attr.TuneResultCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.UB: {model.getAttr(gp.GRB.Attr.UB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VTag: {model.getAttr(gp.GRB.Attr.VTag)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VType: {model.getAttr(gp.GRB.Attr.VType)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarHintPri: {model.getAttr(gp.GRB.Attr.VarHintPri)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarHintVal: {model.getAttr(gp.GRB.Attr.VarHintVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarName: {model.getAttr(gp.GRB.Attr.VarName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarPreStat: {model.getAttr(gp.GRB.Attr.VarPreStat)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.X: {model.getAttr(gp.GRB.Attr.X)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Xn: {model.getAttr(gp.GRB.Attr.Xn)}'))
    '''

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

def write_instance_result(result: InstanceResult):
    solution_text = 'XXXXXX'
    # TO DO
    curr_date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #           data_atual,        caso_teste,          tempo_total,      conclusao,     melhor_resultado,       num_nos_explorados,       limitante_dual,       gap
    line = f'{curr_date_str},{result.instance.name},{result.exec_time:.3},{solution_text},{result.best_result:.3},{result.nb_explored_nodes:.1},{result.best_expected:.3},{result.gap * 100:.3}'

    with open(RESULTS_FILENAME, 'a') as output_file:
        output_file.write(line + '\n')
        output_file.flush()
    
    return

def init_results_file():
    first_line = 'data_atual,caso_teste,tempo_total,conclusao,melhor_resultado,num_nos_explorados,limitante_dual,gap(%)'

    with open(RESULTS_FILENAME, 'a') as output_file:
        output_file.write(first_line + '\n')

    return


def main():
    init_results_file()
    for instance_name in INSTANCE_NAMES:
        print("\n\tInstância: {}\n".format(instance_name))

        instance = read_instance(f"instances/{instance_name}.in")
        model = setup_instance_model(instance)
        result = solve_instance(instance, model)
        write_instance_result(result)


if __name__ == "__main__":
    main()
