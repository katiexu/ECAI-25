import copy
import pennylane as qml
import torch
import torch.nn as nn
from math import pi
import numpy as np
from Arguments import Arguments
import torch.nn.functional as F
args = Arguments()


def gen_arch(change_code, base_code):  # start from 1, not 0
    # arch_code = base_code[1:] * base_code[0]
    n_qubits = base_code[0]
    arch_code = ([i for i in range(2, n_qubits + 1, 1)] + [1]) * base_code[1]
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]

        for i in range(len(change_code)):
            q = change_code[i][0]  # the qubit changed
            for id, t in enumerate(change_code[i][1:]):
                arch_code[q - 1 + id * n_qubits] = t
    return arch_code


def prune_single(change_code):
    single_dict = {}
    single_dict['current_qubit'] = []
    if change_code != None:
        if type(change_code[0]) != type([]):
            change_code = [change_code]
        length = len(change_code[0])
        change_code = np.array(change_code)
        change_qbit = change_code[:, 0] - 1
        change_code = change_code.reshape(-1, length)
        single_dict['current_qubit'] = change_qbit
        j = 0
        for i in change_qbit:
            single_dict['qubit_{}'.format(i)] = change_code[:, 1:][j].reshape(-1, 2).transpose(1, 0)
            j += 1
    return single_dict


def translator(single_code, enta_code, trainable, arch_code, fold=1):
    single_code = qubit_fold(single_code, 0, fold)
    enta_code = qubit_fold(enta_code, 1, fold)
    n_qubits = arch_code[0]
    n_layers = arch_code[1]

    updated_design = {}
    updated_design = prune_single(single_code)
    net = gen_arch(enta_code, arch_code)

    if trainable == 'full' or enta_code == None:
        updated_design['change_qubit'] = None
    else:
        if type(enta_code[0]) != type([]): enta_code = [enta_code]
        updated_design['change_qubit'] = enta_code[-1][0]

    # num of layers
    updated_design['n_layers'] = n_layers

    for layer in range(updated_design['n_layers']):
        # categories of single-qubit parametric gates
        for i in range(n_qubits):
            updated_design['rot' + str(layer) + str(i)] = 'U3'
        # categories and positions of entangled gates
        for j in range(n_qubits):
            if net[j + layer * n_qubits] > 0:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [j, net[j + layer * n_qubits] - 1])
            else:
                updated_design['enta' + str(layer) + str(j)] = ('CU3', [abs(net[j + layer * n_qubits]) - 1, j])

    updated_design['total_gates'] = updated_design['n_layers'] * n_qubits * 2
    return updated_design


def cir_to_matrix(x, y, arch_code, fold=1):
    # x = qubit_fold(x, 0, fold)
    # y = qubit_fold(y, 1, fold)

    qubits = int(arch_code[0] / fold)
    layers = arch_code[1]
    entangle = gen_arch(y, [qubits, layers])
    entangle = np.array([entangle]).reshape(layers, qubits).transpose(1, 0)
    single = np.ones((qubits, 2 * layers))
    # [[1,1,1,1]
    #  [2,2,2,2]
    #  [3,3,3,3]
    #  [0,0,0,0]]

    if x != None:
        if type(x[0]) != type([]):
            x = [x]
        x = np.array(x)
        index = x[:, 0] - 1
        index = [int(index[i]) for i in range(len(index))]
        single[index] = x[:, 1:]
    arch = np.insert(single, [(2 * i) for i in range(1, layers + 1)], entangle, axis=1)
    return arch.transpose(1, 0)


def qubit_fold(jobs, phase, fold=1):
    if fold > 1:
        job_list = []
        for job in jobs:
            q = job[0]
            if phase == 0:
                job_list.append([2 * q] + job[1:])
                job_list.append([2 * q - 1] + job[1:])
            else:
                job_1 = [2 * q]
                job_2 = [2 * q - 1]
                for k in job[1:]:
                    if q < k:
                        job_1.append(2 * k)
                        job_2.append(2 * k - 1)
                    elif q > k:
                        job_1.append(2 * k - 1)
                        job_2.append(2 * k)
                    else:
                        job_1.append(2 * q)
                        job_2.append(2 * q - 1)
                job_list.append(job_1)
                job_list.append(job_2)
    else:
        job_list = jobs
    return job_list


dev = qml.device("lightning.qubit", wires=args.n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_net(x, q_weights_rot, q_weights_enta, **kwargs):
    current_design = kwargs['design']

    for layer in range(current_design['n_layers']):
        # data reuploading
        for j in range(args.n_qubits):
            if not (j in current_design['current_qubit'] and current_design['qubit_{}'.format(j)][0][layer] == 0):
                qml.RY(x[:,j,0], wires=j)
                qml.RZ(x[:,j,1], wires=j)
                qml.RX(x[:,j,2], wires=j)
                qml.RY(x[:,j,3], wires=j)


        for j in range(args.n_qubits):
            if not (j in current_design['current_qubit'] and current_design['qubit_{}'.format(j)][1][layer] == 0):
                idx = j + layer * args.n_qubits
                qml.Rot(*q_weights_rot[idx], wires=j)
                # single-qubit parametric gates and entangled gates
        # Entangling gates
        for j in range(args.n_qubits):
            enta_gate = current_design['enta' + str(layer) + str(j)]
            if enta_gate[1][0] != enta_gate[1][1]:
                idx = j + layer * args.n_qubits
                qml.CRot(*q_weights_enta[idx], wires=enta_gate[1])

    return [qml.expval(qml.PauliZ(i)) for i in range(args.n_qubits)]


class QuantumLayer(nn.Module):
    def __init__(self, arguments, design):
        super(QuantumLayer, self).__init__()
        self.args = arguments
        self.design = design
        self.q_params_rot, self.q_params_enta = nn.ParameterList(), nn.ParameterList()
        for layer in range(self.design['n_layers']):
            for q in range(self.args.n_qubits):
                # 'trainable' option
                if self.design['change_qubit'] is None:
                    rot_trainable = True
                    enta_trainable = True
                elif q == self.design['change_qubit']:
                    rot_trainable = True
                    enta_trainable = True
                else:
                    rot_trainable = False
                    enta_trainable = False
                self.q_params_rot.append(nn.Parameter(pi * torch.rand(3), requires_grad=rot_trainable))
                self.q_params_enta.append(nn.Parameter(pi * torch.rand(3), requires_grad=enta_trainable))

    def forward(self, x_image, n_qubits, task_name):
        x=x_image
        bsz = x.shape[0]
        kernel_size=self.args.kernel
        x = F.avg_pool2d(x, kernel_size)  # 'down_sample_kernel_size' = 6
        x = x.view(bsz, 4, 4).transpose(1,2)
        output = quantum_net(x, self.q_params_rot, self.q_params_enta, design=self.design)
        q_out = torch.stack([output[i] for i in range(len(output))]).float()        # (n_qubits, batch)
        if len(q_out.shape) == 1:
            q_out = q_out.unsqueeze(1)
        q_out = torch.transpose(q_out, 0, 1)    #(batch, n_qubits)
        return q_out


class QNet(nn.Module):
    def __init__(self, arguments, design):
        super(QNet, self).__init__()
        self.args = arguments
        self.design = design
        self.QuantumLayer = QuantumLayer(self.args, self.design)
        self.Regressor = nn.Linear(self.args.n_qubits, 1)

    def forward(self, x_image, n_qubits, task_name):
        exp_val = self.QuantumLayer(x_image, n_qubits, task_name)
        output = torch.tanh(self.Regressor(exp_val).squeeze(dim=1)) * 3
        return output