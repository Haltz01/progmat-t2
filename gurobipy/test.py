import dataclasses
from typing import Dict, List
import gurobipy as gp
from dataclasses import dataclass

from itsdangerous import exc

cases_name = ["d60900", "d201600", "d401600", "d801600", "e60900", "e801600"]

MAX_SECONDS = 3 * 60  # 3min convertido para segundos -> tempo máximo para cada instância
MAX_AGENTS = 80  # número máximo de agentes

VarDict = Dict[int, Dict[int, gp.Var]]


@dataclass
class Instance:
    nb_agents: int
    nb_tasks: int
    profits: List[List[int]]
    capacityReductions: List[List[int]]
    totalCaps: List[int]


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

        return Instance(nb_agents, nb_tasks, profits, capacityReductions, totalCaps)


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
        print(lmbd())
    except Exception as e:
        print("[ERROR] try_print Exception: {}".format(e))
        pass


def solve_instance(instance: Instance, model: gp.Model) -> None:
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
    print("\tValor da função objetivo: {}".format(model.objVal))
    try_pass(lambda: print(f'gp.GRB.Attr.BarIterCount: {model.getAttr(gp.GRB.Attr.BarIterCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BarX: {model.getAttr(gp.GRB.Attr.BarX)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BatchErrorCode: {model.getAttr(gp.GRB.Attr.BatchErrorCode)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BatchErrorMessage: {model.getAttr(gp.GRB.Attr.BatchErrorMessage)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BatchID: {model.getAttr(gp.GRB.Attr.BatchID)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BatchStatus: {model.getAttr(gp.GRB.Attr.BatchStatus)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVio: {model.getAttr(gp.GRB.Attr.BoundSVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVioIndex: {model.getAttr(gp.GRB.Attr.BoundSVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundSVioSum: {model.getAttr(gp.GRB.Attr.BoundSVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundVio: {model.getAttr(gp.GRB.Attr.BoundVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundVioIndex: {model.getAttr(gp.GRB.Attr.BoundVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BoundVioSum: {model.getAttr(gp.GRB.Attr.BoundVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.BranchPriority: {model.getAttr(gp.GRB.Attr.BranchPriority)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.CBasis: {model.getAttr(gp.GRB.Attr.CBasis)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.CTag: {model.getAttr(gp.GRB.Attr.CTag)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ComplVio: {model.getAttr(gp.GRB.Attr.ComplVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ComplVioIndex: {model.getAttr(gp.GRB.Attr.ComplVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ComplVioSum: {model.getAttr(gp.GRB.Attr.ComplVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConcurrentWinMethod: {model.getAttr(gp.GRB.Attr.ConcurrentWinMethod)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrName: {model.getAttr(gp.GRB.Attr.ConstrName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrResidual: {model.getAttr(gp.GRB.Attr.ConstrResidual)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrResidualIndex: {model.getAttr(gp.GRB.Attr.ConstrResidualIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrResidualSum: {model.getAttr(gp.GRB.Attr.ConstrResidualSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSResidual: {model.getAttr(gp.GRB.Attr.ConstrSResidual)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSResidualIndex: {model.getAttr(gp.GRB.Attr.ConstrSResidualIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSResidualSum: {model.getAttr(gp.GRB.Attr.ConstrSResidualSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSVio: {model.getAttr(gp.GRB.Attr.ConstrSVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSVioIndex: {model.getAttr(gp.GRB.Attr.ConstrSVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrSVioSum: {model.getAttr(gp.GRB.Attr.ConstrSVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrVio: {model.getAttr(gp.GRB.Attr.ConstrVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrVioIndex: {model.getAttr(gp.GRB.Attr.ConstrVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ConstrVioSum: {model.getAttr(gp.GRB.Attr.ConstrVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DNumNZs: {model.getAttr(gp.GRB.Attr.DNumNZs)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DStart: {model.getAttr(gp.GRB.Attr.DStart)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualResidual: {model.getAttr(gp.GRB.Attr.DualResidual)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualResidualIndex: {model.getAttr(gp.GRB.Attr.DualResidualIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualResidualSum: {model.getAttr(gp.GRB.Attr.DualResidualSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSResidual: {model.getAttr(gp.GRB.Attr.DualSResidual)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSResidualIndex: {model.getAttr(gp.GRB.Attr.DualSResidualIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSResidualSum: {model.getAttr(gp.GRB.Attr.DualSResidualSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSVio: {model.getAttr(gp.GRB.Attr.DualSVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSVioIndex: {model.getAttr(gp.GRB.Attr.DualSVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualSVioSum: {model.getAttr(gp.GRB.Attr.DualSVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualVio: {model.getAttr(gp.GRB.Attr.DualVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualVioIndex: {model.getAttr(gp.GRB.Attr.DualVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.DualVioSum: {model.getAttr(gp.GRB.Attr.DualVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FarkasDual: {model.getAttr(gp.GRB.Attr.FarkasDual)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FarkasProof: {model.getAttr(gp.GRB.Attr.FarkasProof)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Fingerprint: {model.getAttr(gp.GRB.Attr.Fingerprint)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceError: {model.getAttr(gp.GRB.Attr.FuncPieceError)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceLength: {model.getAttr(gp.GRB.Attr.FuncPieceLength)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieceRatio: {model.getAttr(gp.GRB.Attr.FuncPieceRatio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.FuncPieces: {model.getAttr(gp.GRB.Attr.FuncPieces)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.GenConstrName: {model.getAttr(gp.GRB.Attr.GenConstrName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.GenConstrType: {model.getAttr(gp.GRB.Attr.GenConstrType)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISConstr: {model.getAttr(gp.GRB.Attr.IISConstr)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISConstrForce: {model.getAttr(gp.GRB.Attr.IISConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISGenConstr: {model.getAttr(gp.GRB.Attr.IISGenConstr)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISGenConstrForce: {model.getAttr(gp.GRB.Attr.IISGenConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISLB: {model.getAttr(gp.GRB.Attr.IISLB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISLBForce: {model.getAttr(gp.GRB.Attr.IISLBForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISMinimal: {model.getAttr(gp.GRB.Attr.IISMinimal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISQConstr: {model.getAttr(gp.GRB.Attr.IISQConstr)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISQConstrForce: {model.getAttr(gp.GRB.Attr.IISQConstrForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISSOS: {model.getAttr(gp.GRB.Attr.IISSOS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISSOSForce: {model.getAttr(gp.GRB.Attr.IISSOSForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISUB: {model.getAttr(gp.GRB.Attr.IISUB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IISUBForce: {model.getAttr(gp.GRB.Attr.IISUBForce)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVio: {model.getAttr(gp.GRB.Attr.IntVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVioIndex: {model.getAttr(gp.GRB.Attr.IntVioIndex)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IntVioSum: {model.getAttr(gp.GRB.Attr.IntVioSum)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsMIP: {model.getAttr(gp.GRB.Attr.IsMIP)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsMultiObj: {model.getAttr(gp.GRB.Attr.IsMultiObj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsQCP: {model.getAttr(gp.GRB.Attr.IsQCP)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IsQP: {model.getAttr(gp.GRB.Attr.IsQP)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.IterCount: {model.getAttr(gp.GRB.Attr.IterCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.JobID: {model.getAttr(gp.GRB.Attr.JobID)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Kappa: {model.getAttr(gp.GRB.Attr.Kappa)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.KappaExact: {model.getAttr(gp.GRB.Attr.KappaExact)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.LB: {model.getAttr(gp.GRB.Attr.LB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Lazy: {model.getAttr(gp.GRB.Attr.Lazy)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.LicenseExpiration: {model.getAttr(gp.GRB.Attr.LicenseExpiration)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MIPGap: {model.getAttr(gp.GRB.Attr.MIPGap)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxBound: {model.getAttr(gp.GRB.Attr.MaxBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxCoeff: {model.getAttr(gp.GRB.Attr.MaxCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxObjCoeff: {model.getAttr(gp.GRB.Attr.MaxObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCCoeff: {model.getAttr(gp.GRB.Attr.MaxQCCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCLCoeff: {model.getAttr(gp.GRB.Attr.MaxQCLCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQCRHS: {model.getAttr(gp.GRB.Attr.MaxQCRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxQObjCoeff: {model.getAttr(gp.GRB.Attr.MaxQObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxRHS: {model.getAttr(gp.GRB.Attr.MaxRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MaxVio: {model.getAttr(gp.GRB.Attr.MaxVio)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinBound: {model.getAttr(gp.GRB.Attr.MinBound)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinCoeff: {model.getAttr(gp.GRB.Attr.MinCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinObjCoeff: {model.getAttr(gp.GRB.Attr.MinObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCCoeff: {model.getAttr(gp.GRB.Attr.MinQCCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCLCoeff: {model.getAttr(gp.GRB.Attr.MinQCLCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQCRHS: {model.getAttr(gp.GRB.Attr.MinQCRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinQObjCoeff: {model.getAttr(gp.GRB.Attr.MinQObjCoeff)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.MinRHS: {model.getAttr(gp.GRB.Attr.MinRHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ModelName: {model.getAttr(gp.GRB.Attr.ModelName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ModelSense: {model.getAttr(gp.GRB.Attr.ModelSense)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.NodeCount: {model.getAttr(gp.GRB.Attr.NodeCount)}'))
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
    try_pass(lambda: print(f'gp.GRB.Attr.Obj: {model.getAttr(gp.GRB.Attr.Obj)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.ObjBound: {model.getAttr(gp.GRB.Attr.ObjBound)}'))
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
    try_pass(lambda: print(f'gp.GRB.Attr.ObjVal: {model.getAttr(gp.GRB.Attr.ObjVal)}'))
    # try_print(lambda: # print(f'gp.GRB.Attr.OpenNodeCount: {model.getAttr(gp.GRB.Attr.OpenNodeCount)}'))
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
    try_pass(lambda: print(f'gp.GRB.Attr.RC: {model.getAttr(gp.GRB.Attr.RC)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.RHS: {model.getAttr(gp.GRB.Attr.RHS)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Runtime: {model.getAttr(gp.GRB.Attr.Runtime)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Work: {model.getAttr(gp.GRB.Attr.Work)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SALBLow: {model.getAttr(gp.GRB.Attr.SALBLow)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SALBUp: {model.getAttr(gp.GRB.Attr.SALBUp)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SAObjLow: {model.getAttr(gp.GRB.Attr.SAObjLow)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SAObjUp: {model.getAttr(gp.GRB.Attr.SAObjUp)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SARHSLow: {model.getAttr(gp.GRB.Attr.SARHSLow)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SARHSUp: {model.getAttr(gp.GRB.Attr.SARHSUp)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SAUBLow: {model.getAttr(gp.GRB.Attr.SAUBLow)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.SAUBUp: {model.getAttr(gp.GRB.Attr.SAUBUp)}'))
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
    try_pass(lambda: print(f'gp.GRB.Attr.Status: {model.getAttr(gp.GRB.Attr.Status)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.TuneResultCount: {model.getAttr(gp.GRB.Attr.TuneResultCount)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.UB: {model.getAttr(gp.GRB.Attr.UB)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.UnbdRay: {model.getAttr(gp.GRB.Attr.UnbdRay)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VBasis: {model.getAttr(gp.GRB.Attr.VBasis)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VTag: {model.getAttr(gp.GRB.Attr.VTag)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VType: {model.getAttr(gp.GRB.Attr.VType)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarHintPri: {model.getAttr(gp.GRB.Attr.VarHintPri)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarHintVal: {model.getAttr(gp.GRB.Attr.VarHintVal)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarName: {model.getAttr(gp.GRB.Attr.VarName)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.VarPreStat: {model.getAttr(gp.GRB.Attr.VarPreStat)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.X: {model.getAttr(gp.GRB.Attr.X)}'))
    try_pass(lambda: print(f'gp.GRB.Attr.Xn: {model.getAttr(gp.GRB.Attr.Xn)}'))


    def test_first_instance():
        instance = read_instance(f'instances/{cases_name[0]}.in')

        print("\n\tInstância: {}\n".format(cases_name[0]))

        print(f'\tNúmero de agentes: {instance.nb_agents}')
        print(f'\tNúmero de tarefas: {instance.nb_tasks}')
        print(f'\tÚltimo lucro: {instance.profits[-1][-1]}')
        print(f'\tÚltima redCap: {instance.capacityReductions[-1][-1]}')
        print(f'\tÚltima totalCap: {instance.totalCaps[-1]}')
    
    # model = setup_instance_model(instance)
    # solve_instance(instance, model)

def main():
    # test_first_instance()
    for instance_name in cases_name:
        print('\n\n\n*************************************START**********************************************')
        print('****************************************************************************************')
        print("\n\tInstância: {}\n".format(instance_name))

        instance = read_instance(f"instances/{instance_name}.in")
        model = setup_instance_model(instance)
        solve_instance(instance, model)

        print('\n\n\n****************************************************************************************')
        print('******************************************END********************************************')
        print('\n\n\n')

if __name__ == "__main__":
    main()
