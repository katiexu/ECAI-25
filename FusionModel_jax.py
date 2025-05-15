import pennylane as qml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from model_utils import *
import pickle
from Arguments import Arguments
args = Arguments()




def translator(single_code, enta_code, trainable, arch_code, fold=1):
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

class DressedQuantumCircuitClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        args,
        design,
        learning_rate=args.qlr,
        max_vmap=32,
        jit=True,
        max_steps=100000,
        convergence_interval=200,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
        scaling=1.0,
        random_state=42,
    ):
        r"""
        Dressed quantum circuit from https://arxiv.org/abs/1912.08278. The model consists of the following sequence
            * a single layer fully connected trainable neural network with tanh activation function
            * a parameterised quantum circuit taking the above outputs as input
            * a single layer fully connected trainable neural network taking local expectation values of the above
              circuit as input

        The last neural network maps to two neurons that we take the softmax of to get class probabilities.
        The model is trained via binary cross entropy loss.

        Args:
            n_layers (int): number of layers in the variational part of the circuit.
            learning_rate (float): initial learning rate for gradient descent.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            batch_size (int): Size of batches used for computing parameter updates.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): string specifying the pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (str): the keyword arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): Seed used for pseudorandom number generation.
        """
        # attributes that do not depend on data
        self.loss_fn = None
        self.n_layers = args.n_layers
        self.learning_rate = learning_rate
        self.batch_size = args.batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.scaling = scaling
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.design = design
        self.args=args
        self.max_vmap=max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.n_qubits_ = None
        self.n_features_ = None
        self.scaler = None  # data scaler will be fitted on training data
        self.initialize()

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def input_transform(self, params, x):
        """
        The first neural network that we implment as matrix multiplication.
        """
        x = jnp.matmul(params["input_weights"], x)
        x = jnp.tanh(x) * jnp.pi / 2
        return x

    def output_transform(self, params, x):
        """
        The final neural network
        """
        x = jnp.matmul(params["output_weights"], x)
        return x

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            for layer in range(self.design['n_layers']):
                # data reuploading
                for j in range(self.args.n_qubits):
                    if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][0][layer] == 0):
                        qml.RY(x[j,0], wires=j)
                        qml.RZ(x[j,1], wires=j)
                        qml.RX(x[j,2], wires=j)
                        qml.RY(x[j,3], wires=j)


                for j in range(self.args.n_qubits):
                    if not (j in self.design['current_qubit'] and self.design['qubit_{}'.format(j)][1][layer] == 0):
                        qml.Rot(*params['rot'][layer][j], wires=j)
                        # single-qubit parametric gates and entangled gates
                # Entangling gates
                for j in range(self.args.n_qubits):
                    enta_gate = self.design['enta' + str(layer) + str(j)]
                    if enta_gate[1][0] != enta_gate[1][1]:
                        qml.CRot(*params['enta'][layer][j], wires=enta_gate[1])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.args.n_qubits)]

        self.circuit = circuit

        def dressed_circuit(params, x):
            # x = self.input_transform(params, x)
            x = jnp.array(circuit(params, x)).T
            # x = self.output_transform(params, x)
            return x

        if self.jit:
            dressed_circuit = jax.jit(dressed_circuit)
        self.forward = jax.vmap(dressed_circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        self.n_features_ = 4
        self.classes_ = self.args.digits_of_interest
        self.n_classes_ = len(self.classes_)

        self.n_qubits_ = self.args.n_qubits

        self.initialize_params()
        self.construct_model()

        def loss_fn(params, X, y):
            vals = self.forward(params, X)
            # convert to 0,1 one hot encoded labels
            labels = jax.nn.one_hot(jax.nn.relu(y), self.n_classes_)
            return jnp.mean(optax.softmax_cross_entropy(vals, labels))

        if self.jit:
            self._loss_fn = jax.jit(loss_fn)
        else:
            self._loss_fn=loss_fn
        self.loss_fn = lambda X, y: self._loss_fn(self.params_, X, y)

    def initialize_params(self):
        # initialise the trainable parameters
        rot_weights = (
            jnp.pi
            * jax.random.uniform(
                shape=(self.n_layers, self.n_qubits_, 3), key=self.generate_key()
            )
        )
        enta_weights = (
            jnp.pi
            * jax.random.uniform(
            shape=(self.n_layers, self.n_qubits_, 3), key=self.generate_key()
            )
        )
        self.params_ = {
            "rot": rot_weights,
            "enta": enta_weights,
            # "output_weights": output_weights,
        }

    def fit(self, X,y,epochs):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """
        optimizer = optax.adam

        self.max_steps=X.shape[0]*epochs//self.batch_size
        self.params_ = train(
            self,
            self._loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        return jax.nn.softmax(self.chunked_forward(self.params_, X))

    def draw(self):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        fig, ax = qml.draw_mpl(self.circuit)(self.params_, self.example_x)
        plt.show()

    # 保存参数
    def save_params(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.params_, f)

    # 加载参数
    def load_params(self, filename):
        with open(filename, 'rb') as f:
            self.params_=pickle.load(f)
