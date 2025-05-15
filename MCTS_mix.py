import pickle
import os
import random
import json
import csv
import numpy as np
import torch
from Node import Node, Color
from sklearn.base import BaseEstimator

# from schemes import Scheme, Scheme_eval
from schemes_jax import Scheme, Scheme_eval

from FusionModel import translator
import datetime
from FusionModel import cir_to_matrix 
import time
from sampling import sampling_node
import copy
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from prepare import *
# from draw import plot_2d_array
from Arguments import Arguments
import argparse
import torch.nn as nn
from GVAE_model import is_valid_ops_adj, generate_single_enta, get_proj_mask

class MCTS:
    def __init__(self, search_space, tree_height, fold, arch_code):
        assert type(search_space)    == type([])
        assert len(search_space)     >= 1
        assert type(search_space[0]) == type([])

        self.search_space   = search_space 
        self.ARCH_CODE      = arch_code
        self.ROOT           = None
        self.Cp             = 0.2
        self.nodes          = []
        self.samples        = {}
        self.samples_true   = {}
        self.samples_compact = {}
        self.TASK_QUEUE     = []
        self.DISPATCHED_JOB = {}
        self.mae_list    = []
        self.JOB_COUNTER    = 0
        self.TOTAL_SEND     = 0
        self.TOTAL_RECV     = 0
        self.ITERATION      = 0
        self.MAX_MAEINV     = 0
        self.MAX_SAMPNUM    = 0
        self.sample_nodes   = []
        self.stages         = 0
        self.sampling_num   = 0   
        self.acc_mean       = 0

        self.tree_height    = tree_height

        # initialize a full tree
        total_nodes = 2**tree_height - 1
        for i in range(1, total_nodes + 1):
            is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 0:
                is_good_kid = False
            if (i-1) > 0 and (i-1) % 2 == 1:
                is_good_kid = True

            parent_id = i // 2 - 1
            if parent_id == -1:
                self.nodes.append(Node(tree_height, fold, None, is_good_kid, self.ARCH_CODE, True))
            else:
                self.nodes.append(Node(tree_height, fold, self.nodes[parent_id], is_good_kid, self.ARCH_CODE, False))

        self.ROOT = self.nodes[0]
        self.CURT = self.ROOT
        self.weight = 'init'
        self.best_model_weight = None
        self.explorations = {'phase': 0, 'iteration': 0, 'single':None, 'enta': None, 'regular': [0.001, 0.002, 0.002]}
        self.best = {'acc': 0, 'model':[]}
        self.task_name = ''
        self.history = [[] for i in range(2)]
        self.qubit_used = []
        self.period = 1
        self.fold = fold
        self.performance_per_gate = []

    def init_train(self, numbers=50):
        
        print("\npopulate search space...")
        self.populate_prediction_data()
        print("finished")
        print("\npredict and partition nets in search space...")
        self.predict_nodes()
        self.check_leaf_bags()
        print("finished")
        self.print_tree()

        self.sampling_arch(numbers)
        self.reset_node_data() 

        print("\ncollect " + str(len(self.TASK_QUEUE)) + " nets for initializing MCTS")

    def re_init_tree(self, mode=None):
        
        self.TASK_QUEUE = []
        self.sample_nodes = []        
        
        # strategy = 'base'
        strategy = self.weight
        
        sorted_changes = [k for k, v in sorted(self.samples_compact.items(), key=lambda x: x[1], reverse=True)]
        epochs = 20        
        # pick best 2 and randomly choose one
        random.seed(self.ITERATION)
        
        best_changes = [eval(sorted_changes[i]) for i in range(1)]
        best_change = random.choice(best_changes)
        explicit = (get_list_dimensions(best_change) < 3)
        if explicit:
            self.ROOT.base_code = best_change
            qubits = [code[0] for code in self.ROOT.base_code]
        
            print('Explicit Change: ', best_change)

            if len(best_change[0]) == len(self.explorations['single'][0]):
                best_change_full = self.insert_job(self.explorations['single'], best_change)
                self.explorations['single'] = best_change_full                
            else:
                best_change_full = self.insert_job(self.explorations['enta'], best_change)
                self.explorations['enta'] = best_change_full
            single = self.explorations['single']
            enta = self.explorations['enta']
        else:
            print('Implicit change:', best_change)
            single = best_change[0]
            enta = best_change[1]
            self.explorations['single'] = single
            self.explorations['enta'] = enta
            qubits = []
        best_arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)
        # plot_2d_array(arch)
        design = translator(single, enta, 'full', self.ARCH_CODE, args.fold)
        model_weight = check_file_with_prefix('weights', 'weight_{}_'.format(self.ITERATION))
        if model_weight:            
            best_model, report = Scheme_eval(design, task, model_weight)
            print('Test ACC: ', report['mae'])
        else:
            if debug:
                epochs = 1
            best_model, report = Scheme(design, task, strategy, epochs)
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime('%m-%d-%H')
            if(isinstance(best_model,BaseEstimator)):
                best_model.save_params('weights/weight_{}_{}'.format(self.ITERATION, formatted_time))
                self.weight = best_model.params_
            else:
                torch.save(best_model.state_dict(), 'weights/weight_{}_{}'.format(self.ITERATION, formatted_time))
                self.weight = best_model.state_dict()
        self.samples_true[json.dumps(np.int8(best_arch).tolist())] = report['mae']

        with open('results/{}_fine.csv'.format(self.task_name), 'a+', newline='') as res:
            writer = csv.writer(res)
            metrics = report['mae']
            if not explicit:
                best_change_full = best_change
            writer.writerow([self.ITERATION, best_change_full, metrics, self.performance_per_gate[-1]])
        
        if qubits != []:
            self.history.append(qubits)
        
        self.ROOT.base_code = None               
        self.qubit_used = self.history[-2:]
        
        self.samples_compact = {}
        self.explorations['iteration'] += 1
        arch_last = single + enta

        samples = 20
        if args.strategy == 'mix':
            samples = 10
            with open('search_space/search_space_mnist_single', 'rb') as file:
                self.search_space = pickle.load(file)
        elif args.strategy == 'explicit':
            with open('search_space/search_space_mnist_4', 'rb') as file:       
                self.search_space = pickle.load(file)
        
        if args.strategy in ['explicit', 'mix']:
            # remove last configuration
            for i in range(len(arch_last)):
                try:
                    self.search_space.remove(arch_last[i])
                except ValueError:
                    pass
            self.search_space = [x for x in self.search_space if [x[0]] not in self.qubit_used]        
            self.init_train(samples)

        if args.strategy in ['implicit', 'mix']:    
            # implicit search
            
            self.search_space = []
            self.qubit_used = []
            
            
            self.ROOT.base_code = None        
            self.history = [[] for i in range(2)]
            
            self.weight = self.weight
            # print(Color.BLUE + 'Implicit Switch' + Color.RESET)

            arch_next = self.Langevin_update(best_arch, args.SNR)
            # imp_arch_list = self.projection(arch_next, single, enta)
            for arch in arch_next:
                self.search_space.append(arch)        

            self.init_train(samples)        
            # self.qubit_used = qubits


    def get_grad(self, x):
        
        model = self.ROOT.classifier.model
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        x = self.nodes[0].classifier.arch_to_z([x]).cuda()
        x.requires_grad_(True)
        x.retain_grad()
        n_heads = 3
        y = torch.tensor([[1, 1, 1]]).cuda()

        # clear grads        
        optimizer.zero_grad()

        # forward to get predicted values
        outputs = model(x)
        loss = loss_fn(outputs[0], y.long())
        loss.backward(retain_graph=True)
        return x, x.grad

    def compute_scaling_factor(self, x, decoder, snr_target, d):
            """
            Compute the scaling factor c based on the given formula.
            
            Args:
                x (torch.Tensor): Input tensor.
                decoder (nn.Module): Decoder model.
                snr_target (float): Target signal-to-noise ratio.
                d (int): Dimensionality of the input.

            Returns:
                float: Scaling factor c.
            """
            # Step 1: Compute y = decoder(x)
            x.requires_grad_(True)  # Enable gradient computation for x
            y = decoder(x)
            y = y[0]
            # Step 2: Compute ||y||^2 (mean squared norm of y)
            y_norm_squared = torch.mean(torch.norm(y, dim=-1) ** 2)
            
            # Step 3: Compute Jacobian J of the decoder
            J = []
            for i in range(y.shape[2]):  # Iterate over output dimensions
                grad_outputs = torch.zeros_like(y)
                grad_outputs[:, i] = 1.0  # One-hot vector for each output dimension
                J_i = torch.autograd.grad(y, x, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]
                J.append(J_i)
            J = torch.stack(J, dim=1)  # Stack Jacobian components
            
            # Step 4: Compute ||J||_2^2 (Frobenius norm squared of the Jacobian)
            J_norm_squared = torch.sum(J ** 2)
            
            # Step 5: Compute scaling factor c
            x_norm = torch.norm(x.reshape(x.shape[0], -1), dim=-1).mean()
            c = torch.sqrt(y_norm_squared / (snr_target * d * J_norm_squared)) * (x_norm / torch.sqrt(torch.tensor(d, dtype=torch.float32)))
        
            return c.item() 
    
    def Langevin_update(self, x, snr=10, n_steps=20, step_size=0.01):
        
        x = self.ROOT.classifier.arch_to_z([x])
        x_valid_list = []

        # Compute scaling factor c
        decoder = self.ROOT.classifier.GVAE_model.decoder
        d = x.shape[2]  # Dimensionality
        c = self.compute_scaling_factor(x, decoder, snr, d)
        
        # x_norm_per_sample = torch.norm(x, dim=2, keepdim=True)

        for i in range(1000):
            noise = torch.randn_like(x)
            step_size = c
            x_new = x + step_size * noise
            # x_new = noise * x
            x_new = decoder(x_new)
            mask = get_proj_mask(x_new[0], arch_code[0], arch_code[0])
            if is_valid_ops_adj(x_new[0], int(arch_code[0]/self.fold)):
                gate_matrix = x_new[0] + mask
                single,enta, _ = generate_single_enta(gate_matrix, int(args.n_qubits/args.fold))
                if [single, enta] not in x_valid_list:
                    x_valid_list.append([single, enta])
        print('Number of valid ciruicts:', len(x_valid_list))
        return x_valid_list

    def dump_all_states(self, num_samples):
        node_path = 'states/mcts_agent'
        self.reset_node_data()
        with open(node_path+'_'+str(num_samples), 'wb') as outfile:
            pickle.dump(self, outfile)


    def reset_node_data(self):
        for i in self.nodes:
            i.clear_data()

    def set_arch(self, phase, best_change):
        # if phase == 0:
        if len(best_change[0]) == len(self.explorations['single'][0]):          
            self.explorations['single'] = best_change
            # self.explorations['single'] = None
        else:
            self.explorations['enta'] = best_change
            # self.explorations['enta'] = None

        self.explorations['phase'] = phase        


    def populate_training_data(self):
        self.reset_node_data()
        for k, v in self.samples.items():
            self.ROOT.put_in_bag(json.loads(k), v)

    def populate_prediction_data(self):
        # self.reset_node_data()
        for k in self.search_space:
            self.ROOT.put_in_bag(k, 0.0)


    def train_nodes(self):
        for i in self.nodes:
            i.train(self.tree_height)


    def predict_nodes(self, method = None, dataset =None):
        for i in self.nodes:            
            if dataset:
                i.predict_validation()
            else:
                i.predict(self.explorations, method)


    def check_leaf_bags(self):
        counter = 0
        for i in self.nodes:
            if i.is_leaf is True:
                counter += len(i.bag[0])
        assert counter == len(self.search_space)


    def reset_to_root(self):
        self.CURT = self.ROOT


    def print_tree(self):
        print('\n'+'-'*100)
        for i in self.nodes:
            print(i)
        print('-'*100)


    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        self.ROOT.counter += 1
        while curt_node.is_leaf == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            if torch.rand(1) < curt_node.delta:
                # id = torch.randint(0, len(curt_node.kids), (1,))
                id = np.random.choice(np.argwhere(UCT == np.amin(UCT)).reshape(-1), 1)[0]
            else:
                id = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            curt_node = curt_node.kids[id]
            self.nodes[curt_node.id].counter += 1
        return curt_node
    
    def sampling_arch(self, number=10):
        print('Used Qubits:', self.qubit_used)
        h = 2 ** (self.tree_height-1) - 1
        for i in range(0, number):
            # select
            target_bin   = self.select()           
            qubits = self.qubit_used
            sampled_arch = target_bin.sample_arch(qubits)
            # NOTED: the sampled arch can be None 
            if sampled_arch is not None:                    
                # push the arch into task queue                
                self.TASK_QUEUE.append(sampled_arch)                    
                self.sample_nodes.append(target_bin.id-h)
            else:
                # trail 1: pick a network from the left leaf
                for n in self.nodes:
                    if n.is_leaf == True:
                        sampled_arch = n.sample_arch(qubits)
                        if sampled_arch is not None:
                            # print("\nselected node" + str(n.id-7) + " in leaf layer")                            
                            self.TASK_QUEUE.append(sampled_arch)                                
                            self.sample_nodes.append(n.id-h)
                            break
                        else:
                            continue
            # if type(sampled_arch[0]) == type([]):
            if get_list_dimensions(sampled_arch) == 2:
                arch = sampled_arch[-1]
            else:
                arch = sampled_arch
            self.search_space.remove(arch)        

    def insert_job(self, change_code, job_input):
        job = copy.deepcopy(job_input)
        if type(job[0]) == type([]):
            qubit = [sub[0] for sub in job]
        else:
            qubit = [job[0]]
            job = [job]
        if change_code != None:            
            for change in change_code:
                if change[0] not in qubit:
                    job.append(change)
        return job


    def evaluate_jobs_before(self):
        jobs = []
        designs =[]        
        archs = []
        nodes = []
        while len(self.TASK_QUEUE) > 0:            
           
            job = self.TASK_QUEUE.pop()
            sample_node = self.sample_nodes.pop()
            if type(job[0]) != type([]):
                job = [job]            
            # if self.explorations['phase'] == 0:
            if get_list_dimensions(job) < 3:
                if len(job[0]) == len(self.explorations['single'][0]):
                    single = self.insert_job(self.explorations['single'], job)
                    enta = self.explorations['enta']
                else:
                    single = self.explorations['single']
                    enta = self.insert_job(self.explorations['enta'], job)
            else:
                single = job[0]
                enta = job[1]
            design = translator(single, enta, 'full', self.ARCH_CODE, args.fold)
            arch = cir_to_matrix(single, enta, self.ARCH_CODE, args.fold)
            
            jobs.append(job)
            designs.append(design)
            archs.append(arch)
            nodes.append(sample_node)

        return jobs, designs, archs, nodes

    def evaluate_jobs_after(self, results, jobs, archs, nodes):
        performance_per_gate = []
        for i in range(len(jobs)):
            acc = results[i]
            job = jobs[i]  
            job_str = json.dumps(job)
            arch = archs[i]
            arch_str = json.dumps(np.int8(arch).tolist())
            
            penaly, gates = count_gates(arch, self.explorations['regular'])
            n_gates = gates['uploading'] + gates['single'] + gates['enta'] 
            if regular == True:
                p_acc = acc - penaly
            else:
                p_acc = acc                  
            
            performance_per_gate.append(acc/n_gates)
            self.samples[arch_str] = p_acc
            self.samples_true[arch_str] = acc
            self.samples_compact[job_str] = p_acc
            sample_node = nodes[i]
            print("job:", job_str, "acc:", acc, "p_acc:", p_acc)
            with open('results/{}.csv'.format(self.task_name), 'a+', newline='') as res:
                writer = csv.writer(res)                                        
                num_id = len(self.samples)
                writer.writerow([num_id, job_str, sample_node, acc, p_acc])
            self.mae_list.append(acc)
        self.performance_per_gate.append(np.mean(performance_per_gate))
        print('Performance per gate:', np.mean(performance_per_gate)) 
            

    def pre_search(self, iter):       
        # save current state
        self.ITERATION = iter
        if self.ITERATION > 0:
            self.dump_all_states(self.sampling_num + len(self.samples))
        print("\niteration:", self.ITERATION)
        if self.task_name == 'MOSI':
            period = 5
            number = 50
        else:
            period = 1
            number = 20

        if (self.ITERATION % period == 0): 
            if self.ITERATION == 0:
                self.init_train(number)                    
            else:
                self.re_init_tree()                                        

        # evaluate jobs:
        print("\nevaluate jobs...")
        self.mae_list = []
        jobs, designs, archs, nodes = self.evaluate_jobs_before()

        return jobs, designs, archs, nodes
    
    def post_search(self, jobs, results, archs, nodes):
                
        self.evaluate_jobs_after(results, jobs, archs, nodes)
        print("\nfinished all jobs in task queue")            

        # assemble the training data:
        print("\npopulate training data...")
        self.populate_training_data()
        print("finished")

        # training the tree
        print("\ntrain classifiers in nodes...")
        if torch.cuda.is_available():
            print("using cuda device")
        else:
            print("using cpu device")
        
        start = time.time()
        self.train_nodes()
        print("finished")
        end = time.time()
        print("Running time: %s seconds" % (end - start))
       
        # clear the data in nodes
        self.reset_node_data()                      

        print("\npopulate prediction data...")
        self.populate_prediction_data()
        print("finished")        
        print("\npredict and partition nets in search space...")
        self.predict_nodes() 
        self.check_leaf_bags()
        print("finished")
        print(self.ROOT.delta_history[-1])
        self.print_tree()
        # # sampling nodes
        # # nodes = [0, 1, 2, 3, 8, 12, 13, 14, 15]
        # nodes = [0, 3, 12, 15]
        # sampling_node(self, nodes, dataset, self.ITERATION)
        
        random.seed(self.ITERATION)
        # self.sampling_arch(10)
    
    def projection(self, arch_next, single, enta):
        # Define the projection logic here
        single = sorted(single, key=lambda x: x[0])
        enta = sorted(enta, key=lambda x: x[0])
        single = np.array(single)
        enta = np.array(enta)
        projected_archs = []
        for arch in arch_next:
            new_single = single * (arch[0]==-1) + arch[0] * (arch[0]!=-1)
            new_enta = enta * (arch[1]==-1) + arch[1] * (arch[1]!=-1)
            projected_archs.append([new_single.tolist(), new_enta.tolist()])
        return projected_archs


def Scheme_mp(design, job, task, weight, i, q=None):
    step = len(design)    
    if get_list_dimensions(job[0]) < 3:
        epoch = 1
    else:
        epoch = 2
   
    for j in range(step):
        print('Arch:', job[j])
        _, report = Scheme(design[j], task, weight, epoch, verbs=1)
        q.put([i*step+j, report['mae']])

def count_gates(arch, coeff=None):
    # x = [item for i in [2,3,4,1] for item in [1,1,i]]
    qubits = int(args.n_qubits / args.fold)
    layers = args.n_layers
    x = [[0, 0, i]*4 for i in range(1,qubits+1)] 
    x = np.transpose(x, (1,0))
    x = np.sign(abs(x-arch))
    # if coeff != None:
    #     coeff = np.reshape(coeff * 4, (-1,1))
    #     y = (x * coeff).sum()
    # else:
    #     y = 0
    stat = {}
    stat['uploading'] = x[[3*i for i in range(layers)]].sum()
    stat['single'] = x[[3*i+1 for i in range(layers)]].sum()
    stat['enta'] = x[[3*i+2 for i in range(layers)]].sum()

    y = coeff[0] * stat['single'] + coeff[1] * stat['uploading'] + coeff[2] * stat['enta']

    return y, stat

def analysis_result(samples, ranks):
    gate_stat = []    
    sorted_changes = [k for k, v in sorted(samples.items(), key=lambda x: x[1], reverse=True)]
    for i in range(ranks):
        _, gates = count_gates(eval(sorted_changes[i]))
        gate_stat.append(list(gates.values()))
    mean = np.mean(gate_stat, axis=0)
    return mean

def sampling_qubits(search_space, qubits):
    arch_list = []
    while len(qubits) > 0:    
        arch = random.sample(search_space, 1)
        if arch[0][0] in qubits:
            qubits.remove(arch[0][0])
            arch_list.append(arch[0])
    return arch_list

def create_agent(task, arch_code, pre_file, node=None):
    if files:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(state_path, x)))
        node_path = os.path.join(state_path, files[-1])
        if node: node_path = node        
        with open(node_path, 'rb') as json_data:
            agent = pickle.load(json_data)
        print("\nresume searching,", agent.ITERATION, "iterations completed before")
        print("=====>loads:", len(agent.samples), "samples")        
        print("=====>loads:", len(agent.TASK_QUEUE), 'tasks')
    else:        
        
        with open('search_space/search_space_mnist_4', 'rb') as file:
            search_space = pickle.load(file)

        n_qubits = int(arch_code[0] / args.fold)
        n_layers = arch_code[1]
        
        if task['task'] == 'MNIST-10':
            with open('search_space/search_space_mnist_10', 'rb') as file:
                search_space = pickle.load(file)

        agent = MCTS(search_space, 4, args.fold, arch_code)
        agent.task_name = task['task']+'_'+task['option']

        if pre_file in init_weights:
            if isinstance(agent.nodes[0].classifier.model,BaseEstimator):
                agent.nodes[0].classifier.model.load_params(os.path.join(init_weight_path, pre_file))
            else:
                agent.nodes[0].classifier.model.load_state_dict(torch.load(os.path.join(init_weight_path, pre_file)), strict= True)
       
        # strong entanglement
        # n_qubits = arch_code[0]
        n_layers = arch_code[1]
        
        single = [[i]+[1]*2*n_layers for i in range(1,n_qubits+1)]
        enta = [[i]+[i+1]*n_layers for i in range(1,n_qubits)]+[[n_qubits]+[1]*n_layers]
        
        agent.explorations['single'] = single
        agent.explorations['enta'] = enta
        
        design = translator(single, enta, 'full', arch_code, args.fold)
                
        if args.init_weight in init_weights:
            agent.weight = torch.load(os.path.join(init_weight_path, args.init_weight))
        else:            
            best_model, report = Scheme(design, task, 'init', 30, None, 'save')
            agent.weight = best_model.state_dict()

            with open('results/{}_fine.csv'.format(task['task']+'_'+task['option']), 'a+', newline='') as res:
                writer = csv.writer(res)
                writer.writerow([0, [single, enta], report['mae']]) 
        
    return agent


if __name__ == '__main__':
    
     # set random seed
    random.seed(42)
    np.random.seed(42)
    torch.random.manual_seed(42)

    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('-task', type=str, required=False, default='MNIST', help='Task name, e.g., MNIST or MNIST-10')
    parser.add_argument('-pretrain', type=str, required=False, default='no_pretrain', help='filename of pretraining weights, e.g. pre_weights')

    args_c = parser.parse_args()
    # task = args_c.task
    
    # task = {
    # 'task': 'QML_Linear_32d_mix_reg',
    # 'n_qubits': 8,
    # 'n_layers': 4,
    # 'fold': 2
    # }

    task = {
    'task': 'MNIST_4',
    'option': 'implicit_reg',
    'n_qubits': 4,
    'n_layers': 4,
    'fold': 1
    }

    mp.set_start_method('spawn')

    saved = None
    # saved = 'states/mcts_agent_20'
    num_processes = 2             
    
    check_file(task['task']+'_'+task['option'])
    
    arch_code = [task['n_qubits'], task['n_layers']]
    args = Arguments(**task)
    agent = create_agent(task, arch_code, args_c.pretrain, saved)
    ITERATION = agent.ITERATION
    debug = False
    regular = True


    for iter in range(ITERATION, 30):
        jobs, designs, archs, nodes = agent.pre_search(iter)
        results = {}
        n_jobs = len(jobs)
        step = n_jobs // num_processes
        res = n_jobs % num_processes        
        if not debug:
            with Manager() as manager:
                q = manager.Queue()
                with mp.Pool(processes = num_processes) as pool:        
                    pool.starmap(Scheme_mp, [(designs[i*step : (i+1)*step], jobs[i*step : (i+1)*step], task, agent.weight, i, q) for i in range(num_processes)])            
                    pool.starmap(Scheme_mp, [(designs[n_jobs-i-1 : n_jobs-i], jobs[i*step : (i+1)*step], task, agent.weight, n_jobs-i-1, q) for i in range(res)])
                while not q.empty():
                    [i, acc] = q.get()
                    results[i] = acc
        else:
            for i in range(n_jobs):
                results[i] = random.uniform(0.75, 0.8)

        agent.post_search(jobs, results, archs, nodes)

    print('The best model: ', agent.best['acc'])
    agent.dump_all_states(agent.sampling_num + len(agent.samples))
    # plot_2d_array(agent.best['model'])
    
    Range = [0.8, 0.82]
    rank = 20

    print('<{}:'.format(Range[0]), sum(value < Range[0] for value in list(agent.samples_true.values())))
    print('({}, {}):'.format(Range[0], Range[1]), sum((value in Range)  for value in list(agent.samples_true.values())))
    print('>{}:'.format(Range[1]), sum(value > Range[1]  for value in list(agent.samples_true.values())))
    # print('Gate numbers of top {}: {}'.format(rank, analysis_result(agent.samples_true, rank)))