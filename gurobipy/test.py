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

    try:
        print()
        print("\tValor da função objetivo: {}".format(model.objVal))
        print(f'gp.GRB.Attr.BarIterCount: {model.getAttr(gp.GRB.Attr.BarIterCount)}')
        print(f'gp.GRB.Attr.BarX: {model.getAttr(gp.GRB.Attr.BarX)}')
        print(f'gp.GRB.Attr.BatchErrorCode: {model.getAttr(gp.GRB.Attr.BatchErrorCode)}')
        print(f'gp.GRB.Attr.BatchErrorMessage: {model.getAttr(gp.GRB.Attr.BatchErrorMessage)}')
        print(f'gp.GRB.Attr.BatchID: {model.getAttr(gp.GRB.Attr.BatchID)}')
        print(f'gp.GRB.Attr.BatchStatus: {model.getAttr(gp.GRB.Attr.BatchStatus)}')
        print(f'gp.GRB.Attr.BoundSVio: {model.getAttr(gp.GRB.Attr.BoundSVio)}')
        print(f'gp.GRB.Attr.BoundSVioIndex: {model.getAttr(gp.GRB.Attr.BoundSVioIndex)}')
        print(f'gp.GRB.Attr.BoundSVioSum: {model.getAttr(gp.GRB.Attr.BoundSVioSum)}')
        print(f'gp.GRB.Attr.BoundVio: {model.getAttr(gp.GRB.Attr.BoundVio)}')
        print(f'gp.GRB.Attr.BoundVioIndex: {model.getAttr(gp.GRB.Attr.BoundVioIndex)}')
        print(f'gp.GRB.Attr.BoundVioSum: {model.getAttr(gp.GRB.Attr.BoundVioSum)}')
        print(f'gp.GRB.Attr.BranchPriority: {model.getAttr(gp.GRB.Attr.BranchPriority)}')
        print(f'gp.GRB.Attr.CBasis: {model.getAttr(gp.GRB.Attr.CBasis)}')
        print(f'gp.GRB.Attr.CTag: {model.getAttr(gp.GRB.Attr.CTag)}')
        print(f'gp.GRB.Attr.ComplVio: {model.getAttr(gp.GRB.Attr.ComplVio)}')
        print(f'gp.GRB.Attr.ComplVioIndex: {model.getAttr(gp.GRB.Attr.ComplVioIndex)}')
        print(f'gp.GRB.Attr.ComplVioSum: {model.getAttr(gp.GRB.Attr.ComplVioSum)}')
        print(f'gp.GRB.Attr.ConcurrentWinMethod: {model.getAttr(gp.GRB.Attr.ConcurrentWinMethod)}')
        print(f'gp.GRB.Attr.ConstrName: {model.getAttr(gp.GRB.Attr.ConstrName)}')
        print(f'gp.GRB.Attr.ConstrResidual: {model.getAttr(gp.GRB.Attr.ConstrResidual)}')
        print(f'gp.GRB.Attr.ConstrResidualIndex: {model.getAttr(gp.GRB.Attr.ConstrResidualIndex)}')
        print(f'gp.GRB.Attr.ConstrResidualSum: {model.getAttr(gp.GRB.Attr.ConstrResidualSum)}')
        print(f'gp.GRB.Attr.ConstrSResidual: {model.getAttr(gp.GRB.Attr.ConstrSResidual)}')
        print(f'gp.GRB.Attr.ConstrSResidualIndex: {model.getAttr(gp.GRB.Attr.ConstrSResidualIndex)}')
        print(f'gp.GRB.Attr.ConstrSResidualSum: {model.getAttr(gp.GRB.Attr.ConstrSResidualSum)}')
        print(f'gp.GRB.Attr.ConstrSVio: {model.getAttr(gp.GRB.Attr.ConstrSVio)}')
        print(f'gp.GRB.Attr.ConstrSVioIndex: {model.getAttr(gp.GRB.Attr.ConstrSVioIndex)}')
        print(f'gp.GRB.Attr.ConstrSVioSum: {model.getAttr(gp.GRB.Attr.ConstrSVioSum)}')
        print(f'gp.GRB.Attr.ConstrVio: {model.getAttr(gp.GRB.Attr.ConstrVio)}')
        print(f'gp.GRB.Attr.ConstrVioIndex: {model.getAttr(gp.GRB.Attr.ConstrVioIndex)}')
        print(f'gp.GRB.Attr.ConstrVioSum: {model.getAttr(gp.GRB.Attr.ConstrVioSum)}')
        print(f'gp.GRB.Attr.DNumNZs: {model.getAttr(gp.GRB.Attr.DNumNZs)}')
        print(f'gp.GRB.Attr.DStart: {model.getAttr(gp.GRB.Attr.DStart)}')
        print(f'gp.GRB.Attr.DualResidual: {model.getAttr(gp.GRB.Attr.DualResidual)}')
        print(f'gp.GRB.Attr.DualResidualIndex: {model.getAttr(gp.GRB.Attr.DualResidualIndex)}')
        print(f'gp.GRB.Attr.DualResidualSum: {model.getAttr(gp.GRB.Attr.DualResidualSum)}')
        print(f'gp.GRB.Attr.DualSResidual: {model.getAttr(gp.GRB.Attr.DualSResidual)}')
        print(f'gp.GRB.Attr.DualSResidualIndex: {model.getAttr(gp.GRB.Attr.DualSResidualIndex)}')
        print(f'gp.GRB.Attr.DualSResidualSum: {model.getAttr(gp.GRB.Attr.DualSResidualSum)}')
        print(f'gp.GRB.Attr.DualSVio: {model.getAttr(gp.GRB.Attr.DualSVio)}')
        print(f'gp.GRB.Attr.DualSVioIndex: {model.getAttr(gp.GRB.Attr.DualSVioIndex)}')
        print(f'gp.GRB.Attr.DualSVioSum: {model.getAttr(gp.GRB.Attr.DualSVioSum)}')
        print(f'gp.GRB.Attr.DualVio: {model.getAttr(gp.GRB.Attr.DualVio)}')
        print(f'gp.GRB.Attr.DualVioIndex: {model.getAttr(gp.GRB.Attr.DualVioIndex)}')
        print(f'gp.GRB.Attr.DualVioSum: {model.getAttr(gp.GRB.Attr.DualVioSum)}')
        print(f'gp.GRB.Attr.FarkasDual: {model.getAttr(gp.GRB.Attr.FarkasDual)}')
        print(f'gp.GRB.Attr.FarkasProof: {model.getAttr(gp.GRB.Attr.FarkasProof)}')
        print(f'gp.GRB.Attr.Fingerprint: {model.getAttr(gp.GRB.Attr.Fingerprint)}')
        print(f'gp.GRB.Attr.FuncPieceError: {model.getAttr(gp.GRB.Attr.FuncPieceError)}')
        print(f'gp.GRB.Attr.FuncPieceLength: {model.getAttr(gp.GRB.Attr.FuncPieceLength)}')
        print(f'gp.GRB.Attr.FuncPieceRatio: {model.getAttr(gp.GRB.Attr.FuncPieceRatio)}')
        print(f'gp.GRB.Attr.FuncPieces: {model.getAttr(gp.GRB.Attr.FuncPieces)}')
        print(f'gp.GRB.Attr.GenConstrName: {model.getAttr(gp.GRB.Attr.GenConstrName)}')
        print(f'gp.GRB.Attr.GenConstrType: {model.getAttr(gp.GRB.Attr.GenConstrType)}')
        print(f'gp.GRB.Attr.IISConstr: {model.getAttr(gp.GRB.Attr.IISConstr)}')
        print(f'gp.GRB.Attr.IISConstrForce: {model.getAttr(gp.GRB.Attr.IISConstrForce)}')
        print(f'gp.GRB.Attr.IISGenConstr: {model.getAttr(gp.GRB.Attr.IISGenConstr)}')
        print(f'gp.GRB.Attr.IISGenConstrForce: {model.getAttr(gp.GRB.Attr.IISGenConstrForce)}')
        print(f'gp.GRB.Attr.IISLB: {model.getAttr(gp.GRB.Attr.IISLB)}')
        print(f'gp.GRB.Attr.IISLBForce: {model.getAttr(gp.GRB.Attr.IISLBForce)}')
        print(f'gp.GRB.Attr.IISMinimal: {model.getAttr(gp.GRB.Attr.IISMinimal)}')
        print(f'gp.GRB.Attr.IISQConstr: {model.getAttr(gp.GRB.Attr.IISQConstr)}')
        print(f'gp.GRB.Attr.IISQConstrForce: {model.getAttr(gp.GRB.Attr.IISQConstrForce)}')
        print(f'gp.GRB.Attr.IISSOS: {model.getAttr(gp.GRB.Attr.IISSOS)}')
        print(f'gp.GRB.Attr.IISSOSForce: {model.getAttr(gp.GRB.Attr.IISSOSForce)}')
        print(f'gp.GRB.Attr.IISUB: {model.getAttr(gp.GRB.Attr.IISUB)}')
        print(f'gp.GRB.Attr.IISUBForce: {model.getAttr(gp.GRB.Attr.IISUBForce)}')
        print(f'gp.GRB.Attr.IntVio: {model.getAttr(gp.GRB.Attr.IntVio)}')
        print(f'gp.GRB.Attr.IntVioIndex: {model.getAttr(gp.GRB.Attr.IntVioIndex)}')
        print(f'gp.GRB.Attr.IntVioSum: {model.getAttr(gp.GRB.Attr.IntVioSum)}')
        print(f'gp.GRB.Attr.IsMIP: {model.getAttr(gp.GRB.Attr.IsMIP)}')
        print(f'gp.GRB.Attr.IsMultiObj: {model.getAttr(gp.GRB.Attr.IsMultiObj)}')
        print(f'gp.GRB.Attr.IsQCP: {model.getAttr(gp.GRB.Attr.IsQCP)}')
        print(f'gp.GRB.Attr.IsQP: {model.getAttr(gp.GRB.Attr.IsQP)}')
        print(f'gp.GRB.Attr.IterCount: {model.getAttr(gp.GRB.Attr.IterCount)}')
        print(f'gp.GRB.Attr.JobID: {model.getAttr(gp.GRB.Attr.JobID)}')
        print(f'gp.GRB.Attr.Kappa: {model.getAttr(gp.GRB.Attr.Kappa)}')
        print(f'gp.GRB.Attr.KappaExact: {model.getAttr(gp.GRB.Attr.KappaExact)}')
        print(f'gp.GRB.Attr.LB: {model.getAttr(gp.GRB.Attr.LB)}')
        print(f'gp.GRB.Attr.Lazy: {model.getAttr(gp.GRB.Attr.Lazy)}')
        print(f'gp.GRB.Attr.LicenseExpiration: {model.getAttr(gp.GRB.Attr.LicenseExpiration)}')
        print(f'gp.GRB.Attr.MIPGap: {model.getAttr(gp.GRB.Attr.MIPGap)}')
        print(f'gp.GRB.Attr.MaxBound: {model.getAttr(gp.GRB.Attr.MaxBound)}')
        print(f'gp.GRB.Attr.MaxCoeff: {model.getAttr(gp.GRB.Attr.MaxCoeff)}')
        print(f'gp.GRB.Attr.MaxObjCoeff: {model.getAttr(gp.GRB.Attr.MaxObjCoeff)}')
        print(f'gp.GRB.Attr.MaxQCCoeff: {model.getAttr(gp.GRB.Attr.MaxQCCoeff)}')
        print(f'gp.GRB.Attr.MaxQCLCoeff: {model.getAttr(gp.GRB.Attr.MaxQCLCoeff)}')
        print(f'gp.GRB.Attr.MaxQCRHS: {model.getAttr(gp.GRB.Attr.MaxQCRHS)}')
        print(f'gp.GRB.Attr.MaxQObjCoeff: {model.getAttr(gp.GRB.Attr.MaxQObjCoeff)}')
        print(f'gp.GRB.Attr.MaxRHS: {model.getAttr(gp.GRB.Attr.MaxRHS)}')
        print(f'gp.GRB.Attr.MaxVio: {model.getAttr(gp.GRB.Attr.MaxVio)}')
        print(f'gp.GRB.Attr.MinBound: {model.getAttr(gp.GRB.Attr.MinBound)}')
        print(f'gp.GRB.Attr.MinCoeff: {model.getAttr(gp.GRB.Attr.MinCoeff)}')
        print(f'gp.GRB.Attr.MinObjCoeff: {model.getAttr(gp.GRB.Attr.MinObjCoeff)}')
        print(f'gp.GRB.Attr.MinQCCoeff: {model.getAttr(gp.GRB.Attr.MinQCCoeff)}')
        print(f'gp.GRB.Attr.MinQCLCoeff: {model.getAttr(gp.GRB.Attr.MinQCLCoeff)}')
        print(f'gp.GRB.Attr.MinQCRHS: {model.getAttr(gp.GRB.Attr.MinQCRHS)}')
        print(f'gp.GRB.Attr.MinQObjCoeff: {model.getAttr(gp.GRB.Attr.MinQObjCoeff)}')
        print(f'gp.GRB.Attr.MinRHS: {model.getAttr(gp.GRB.Attr.MinRHS)}')
        print(f'gp.GRB.Attr.ModelName: {model.getAttr(gp.GRB.Attr.ModelName)}')
        print(f'gp.GRB.Attr.ModelSense: {model.getAttr(gp.GRB.Attr.ModelSense)}')
        print(f'gp.GRB.Attr.NodeCount: {model.getAttr(gp.GRB.Attr.NodeCount)}')
        print(f'gp.GRB.Attr.NumBinVars: {model.getAttr(gp.GRB.Attr.NumBinVars)}')
        print(f'gp.GRB.Attr.NumConstrs: {model.getAttr(gp.GRB.Attr.NumConstrs)}')
        print(f'gp.GRB.Attr.NumGenConstrs: {model.getAttr(gp.GRB.Attr.NumGenConstrs)}')
        print(f'gp.GRB.Attr.NumIntVars: {model.getAttr(gp.GRB.Attr.NumIntVars)}')
        print(f'gp.GRB.Attr.NumNZs: {model.getAttr(gp.GRB.Attr.NumNZs)}')
        print(f'gp.GRB.Attr.NumObj: {model.getAttr(gp.GRB.Attr.NumObj)}')
        print(f'gp.GRB.Attr.NumPWLObjVars: {model.getAttr(gp.GRB.Attr.NumPWLObjVars)}')
        print(f'gp.GRB.Attr.NumQCNZs: {model.getAttr(gp.GRB.Attr.NumQCNZs)}')
        print(f'gp.GRB.Attr.NumQConstrs: {model.getAttr(gp.GRB.Attr.NumQConstrs)}')
        print(f'gp.GRB.Attr.NumQNZs: {model.getAttr(gp.GRB.Attr.NumQNZs)}')
        print(f'gp.GRB.Attr.NumSOS: {model.getAttr(gp.GRB.Attr.NumSOS)}')
        print(f'gp.GRB.Attr.NumScenarios: {model.getAttr(gp.GRB.Attr.NumScenarios)}')
        print(f'gp.GRB.Attr.NumStart: {model.getAttr(gp.GRB.Attr.NumStart)}')
        print(f'gp.GRB.Attr.NumVars: {model.getAttr(gp.GRB.Attr.NumVars)}')
        print(f'gp.GRB.Attr.Obj: {model.getAttr(gp.GRB.Attr.Obj)}')
        print(f'gp.GRB.Attr.ObjBound: {model.getAttr(gp.GRB.Attr.ObjBound)}')
        print(f'gp.GRB.Attr.ObjBoundC: {model.getAttr(gp.GRB.Attr.ObjBoundC)}')
        print(f'gp.GRB.Attr.ObjCon: {model.getAttr(gp.GRB.Attr.ObjCon)}')
        print(f'gp.GRB.Attr.ObjN: {model.getAttr(gp.GRB.Attr.ObjN)}')
        print(f'gp.GRB.Attr.ObjNAbsTol: {model.getAttr(gp.GRB.Attr.ObjNAbsTol)}')
        print(f'gp.GRB.Attr.ObjNCon: {model.getAttr(gp.GRB.Attr.ObjNCon)}')
        print(f'gp.GRB.Attr.ObjNName: {model.getAttr(gp.GRB.Attr.ObjNName)}')
        print(f'gp.GRB.Attr.ObjNPriority: {model.getAttr(gp.GRB.Attr.ObjNPriority)}')
        print(f'gp.GRB.Attr.ObjNRelTol: {model.getAttr(gp.GRB.Attr.ObjNRelTol)}')
        print(f'gp.GRB.Attr.ObjNVal: {model.getAttr(gp.GRB.Attr.ObjNVal)}')
        print(f'gp.GRB.Attr.ObjNWeight: {model.getAttr(gp.GRB.Attr.ObjNWeight)}')
        print(f'gp.GRB.Attr.ObjVal: {model.getAttr(gp.GRB.Attr.ObjVal)}')
        # print(f'gp.GRB.Attr.OpenNodeCount: {model.getAttr(gp.GRB.Attr.OpenNodeCount)}')
        print(f'gp.GRB.Attr.PStart: {model.getAttr(gp.GRB.Attr.PStart)}')
        print(f'gp.GRB.Attr.PWLObjCvx: {model.getAttr(gp.GRB.Attr.PWLObjCvx)}')
        print(f'gp.GRB.Attr.Partition: {model.getAttr(gp.GRB.Attr.Partition)}')
        print(f'gp.GRB.Attr.Pi: {model.getAttr(gp.GRB.Attr.Pi)}')
        print(f'gp.GRB.Attr.PoolIgnore: {model.getAttr(gp.GRB.Attr.PoolIgnore)}')
        print(f'gp.GRB.Attr.PoolObjBound: {model.getAttr(gp.GRB.Attr.PoolObjBound)}')
        print(f'gp.GRB.Attr.PoolObjVal: {model.getAttr(gp.GRB.Attr.PoolObjVal)}')
        print(f'gp.GRB.Attr.PreFixVal: {model.getAttr(gp.GRB.Attr.PreFixVal)}')
        print(f'gp.GRB.Attr.QCName: {model.getAttr(gp.GRB.Attr.QCName)}')
        print(f'gp.GRB.Attr.QCPi: {model.getAttr(gp.GRB.Attr.QCPi)}')
        print(f'gp.GRB.Attr.QCRHS: {model.getAttr(gp.GRB.Attr.QCRHS)}')
        print(f'gp.GRB.Attr.QCSense: {model.getAttr(gp.GRB.Attr.QCSense)}')
        print(f'gp.GRB.Attr.QCSlack: {model.getAttr(gp.GRB.Attr.QCSlack)}')
        print(f'gp.GRB.Attr.QCTag: {model.getAttr(gp.GRB.Attr.QCTag)}')
        print(f'gp.GRB.Attr.RC: {model.getAttr(gp.GRB.Attr.RC)}')
        print(f'gp.GRB.Attr.RHS: {model.getAttr(gp.GRB.Attr.RHS)}')
        print(f'gp.GRB.Attr.Runtime: {model.getAttr(gp.GRB.Attr.Runtime)}')
        print(f'gp.GRB.Attr.Work: {model.getAttr(gp.GRB.Attr.Work)}')
        print(f'gp.GRB.Attr.SALBLow: {model.getAttr(gp.GRB.Attr.SALBLow)}')
        print(f'gp.GRB.Attr.SALBUp: {model.getAttr(gp.GRB.Attr.SALBUp)}')
        print(f'gp.GRB.Attr.SAObjLow: {model.getAttr(gp.GRB.Attr.SAObjLow)}')
        print(f'gp.GRB.Attr.SAObjUp: {model.getAttr(gp.GRB.Attr.SAObjUp)}')
        print(f'gp.GRB.Attr.SARHSLow: {model.getAttr(gp.GRB.Attr.SARHSLow)}')
        print(f'gp.GRB.Attr.SARHSUp: {model.getAttr(gp.GRB.Attr.SARHSUp)}')
        print(f'gp.GRB.Attr.SAUBLow: {model.getAttr(gp.GRB.Attr.SAUBLow)}')
        print(f'gp.GRB.Attr.SAUBUp: {model.getAttr(gp.GRB.Attr.SAUBUp)}')
        print(f'gp.GRB.Attr.ScenNLB: {model.getAttr(gp.GRB.Attr.ScenNLB)}')
        print(f'gp.GRB.Attr.ScenNName: {model.getAttr(gp.GRB.Attr.ScenNName)}')
        print(f'gp.GRB.Attr.ScenNObj: {model.getAttr(gp.GRB.Attr.ScenNObj)}')
        print(f'gp.GRB.Attr.ScenNObjBound: {model.getAttr(gp.GRB.Attr.ScenNObjBound)}')
        print(f'gp.GRB.Attr.ScenNObjVal: {model.getAttr(gp.GRB.Attr.ScenNObjVal)}')
        print(f'gp.GRB.Attr.ScenNRHS: {model.getAttr(gp.GRB.Attr.ScenNRHS)}')
        print(f'gp.GRB.Attr.ScenNUB: {model.getAttr(gp.GRB.Attr.ScenNUB)}')
        print(f'gp.GRB.Attr.ScenNX: {model.getAttr(gp.GRB.Attr.ScenNX)}')
        print(f'gp.GRB.Attr.Sense: {model.getAttr(gp.GRB.Attr.Sense)}')
        print(f'gp.GRB.Attr.Server: {model.getAttr(gp.GRB.Attr.Server)}')
        print(f'gp.GRB.Attr.Slack: {model.getAttr(gp.GRB.Attr.Slack)}')
        print(f'gp.GRB.Attr.SolCount: {model.getAttr(gp.GRB.Attr.SolCount)}')
        print(f'gp.GRB.Attr.Start: {model.getAttr(gp.GRB.Attr.Start)}')
        print(f'gp.GRB.Attr.Status: {model.getAttr(gp.GRB.Attr.Status)}')
        print(f'gp.GRB.Attr.TuneResultCount: {model.getAttr(gp.GRB.Attr.TuneResultCount)}')
        print(f'gp.GRB.Attr.UB: {model.getAttr(gp.GRB.Attr.UB)}')
        print(f'gp.GRB.Attr.UnbdRay: {model.getAttr(gp.GRB.Attr.UnbdRay)}')
        print(f'gp.GRB.Attr.VBasis: {model.getAttr(gp.GRB.Attr.VBasis)}')
        print(f'gp.GRB.Attr.VTag: {model.getAttr(gp.GRB.Attr.VTag)}')
        print(f'gp.GRB.Attr.VType: {model.getAttr(gp.GRB.Attr.VType)}')
        print(f'gp.GRB.Attr.VarHintPri: {model.getAttr(gp.GRB.Attr.VarHintPri)}')
        print(f'gp.GRB.Attr.VarHintVal: {model.getAttr(gp.GRB.Attr.VarHintVal)}')
        print(f'gp.GRB.Attr.VarName: {model.getAttr(gp.GRB.Attr.VarName)}')
        print(f'gp.GRB.Attr.VarPreStat: {model.getAttr(gp.GRB.Attr.VarPreStat)}')
        print(f'gp.GRB.Attr.X: {model.getAttr(gp.GRB.Attr.X)}')
        print(f'gp.GRB.Attr.Xn: {model.getAttr(gp.GRB.Attr.Xn)}')

    except Exception as e:
        print(e)

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
