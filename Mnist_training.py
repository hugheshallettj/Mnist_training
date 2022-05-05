
# %% Load Libraries

from threading import local
import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn 
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import scipy
import time

rng = np.random.RandomState(123)
tf.random.set_seed(42)

# %% Utility Functions


class Stopwatch():
    """Class for a stopwatch timer.
    """

    def __init__(self):
        self.current_time = 0

    def start(self):
        self.start_time = time.time()
        
    def stop(self):
        self.current_time += time.time()-self.start_time

    def reset(self):
        self.current_time = 0 
        pass    

    def read_time(self):
        return self.current_time


def load_mnist(y_hot_encoding=True):

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    N = len(y_train)

    # preprocessing
    x_train = x_train.reshape(60000, 784)/255
    x_test = x_test.reshape(10000, 784)/255
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    if y_hot_encoding==True:
        C = len(unique_labels(y_train))
        y_hot = np.zeros((N, C))
        
        y_hot = OneHotEncoder(handle_unknown='ignore').fit_transform(y_train).toarray()
        return x_train, x_test, y_hot, y_test
    else:
        return x_train, x_test, y_train, y_test


# %% Make function for training Sparse models on MNIST dataset

def train_mnist(
    model,
    num_ind_points,
    num_iterations,
    inducing_point_locations="Random",
    train_ind_points=True,
    likelihood_func="Gaussian",
    learning_rate_adam=0.1,
    minibatch_size=None,
    initial_kernel_variance=1,
    initial_kernel_lengthscale=10,
    initial_likelihood_hyperparameter=1
    ):
    """Trains sparse GP models on MNIST dataset.

    Args:
        model (str): model with which to train on MNIST dataset
        num_ind_points (int): Number of inducing points to be used in the model.
        num_iterations (int): Number of iterations during optimisation.
        train_ind_points (bool, optional): Set inducing points to be a trainable parameter. Defaults to True.
        likelihood_func (str, optional): Likelihood function to use for GP model. Defaults to Gaussian ("RobustMax" also available).
        learning_rate_adam (float, optional): Learning rate to use in Adam optimiser. Defaults to 0.01.
        minibatch_size (int, optional): Size of minibatch to use during optimisation. Defaults to None (full batch training).

    Returns:
        model (GPflow model object)
        times (array)
        elbo_logs (array)
        accuracy_logs (array)
    """

    if not model == "SGPR" and not model == "SVGP":
        raise ValueError("Model must be either SGPR or SVGP.")
    if not type(num_ind_points) is int:
        raise ValueError("Number of inducing points must be an integer.")
    if not type(train_ind_points) is bool:
        raise ValueError("train_ind_points must be boolean.")
    if not type(num_iterations) is int:
        raise ValueError("Iterations must be an integer.")
    if minibatch_size is not None and not type(minibatch_size) is int:
        raise ValueError("Minibatch size must be an integer.")
    if type(inducing_point_locations) != str and not len(inducing_point_locations)==num_ind_points:
        raise ValueError("Number of inducing points must match length of inducing locations array")


    y_hot_encoding = likelihood_func!="RobustMax"

    x_train, x_test, y_train, y_test = load_mnist( y_hot_encoding=y_hot_encoding)

    data=(x_train, y_train)

    N = len(y_train)

    #* Number of inducing points
    M = num_ind_points

    #* Initialisation of inducing points
    if inducing_point_locations=="Random":
        Z = x_train[:M, :].copy()
    else:
        Z = inducing_point_locations

    #* Number of classes for a classification dataset
    C = len(unique_labels(y_train))

    #* Likelihood
    if likelihood_func=="RobustMax":
        invlink = gpflow.likelihoods.RobustMax(C)  # Robustmax inverse link function
        likelihood = gpflow.likelihoods.MultiClass(C, invlink=invlink)  # Multiclass likelihood
    elif likelihood_func=="Gaussian":
        likelihood = gpflow.likelihoods.Gaussian()

    #* Kernel
    kernel = gpflow.kernels.SquaredExponential()

    #* Initialise the model
    if model=="SVGP":
        m = gpflow.models.SVGP(
            inducing_variable=Z,
            kernel=kernel,
            num_latent_gps=C,
            likelihood=likelihood,
            )
    elif model=="SGPR":
        m = gpflow.models.SGPR(
            data=data,
            inducing_variable=Z,
            kernel=kernel,
            )

    #* Pick value for jitter
    jitter = 1e-3
    gpflow.config.set_default_jitter(jitter)

    #* Initialise hyperparams
    if likelihood_func=="Gaussian":
        m.likelihood.variance.assign(initial_likelihood_hyperparameter)
    # if likelihood_func=="RobustMax":
    #     m.likelihood.invlink.epsilon.assign(initial_likelihood_hyperparameter)
    m.kernel.variance.assign(initial_kernel_variance)
    m.kernel.lengthscales.assign(initial_kernel_lengthscale)

    #* Set trainable parameters
    gpflow.set_trainable(m.inducing_variable, train_ind_points)
    gpflow.set_trainable(m.kernel, train_ind_points)
    gpflow.set_trainable(m.likelihood, train_ind_points)

    gpflow.utilities.print_summary(m, fmt="notebook")

    #* Optimiser object
    opt = tf.optimizers.Adam(learning_rate=learning_rate_adam)

    #* List for ELBO values during 
    elbo_logs = []
    accuracy_logs = []
    times = []

    #* Stopwatch instance
    watch = Stopwatch()
    watch.reset()

    #* Number of logs to make during optimisation
    n_logs = 200

    def log_opt():
        """
        Utility function to log the elbo score for each iteration during optimisaiton.

        Args:
            x
        """
        watch.stop()
        if model == "SGPR":
            ELBO =  m.elbo()
        elif model=="SVGP":
            ELBO =  m.elbo(data)
        elbo_logs.append(float(ELBO))
        times.append(watch.read_time())
        y_pred, var = m.predict_y(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1).reshape(-1, 1)
        accuracy_logs.append(accuracy_score(y_test, y_pred_labels))
        watch.start()
        
        #// logs.append(float(m.log_marginal_likelihood()))


    if model=="SGPR":
        training_loss=m.training_loss

    if model=="SVGP" and minibatch_size is None:
        training_loss= m.training_loss_closure(data)
    elif model=="SVGP" and minibatch_size is not None:
        train_dataset = tf.data.Dataset.from_tensor_slices((data)).repeat().shuffle(N)
        train_iter = iter(train_dataset.batch(minibatch_size))
        training_loss = m.training_loss_closure(train_iter, compile=True)

    def run_adam(iterations):
        for i in range(iterations):
            opt.minimize(
                training_loss,
                m.trainable_variables,
                )
            freq = iterations//n_logs 
            if i % freq == 0:  #-- only log every freq iterations
                log_opt()

    watch.start()  # --Start the watch


    #* Optimise
    run_adam(num_iterations)


    watch.stop() # --Stop the watch for the final time
    print("Model: {}".format(model))
    print("Number of inducing points: {}".format(num_ind_points))
    print("Total time for optimisation: {} seconds".format(watch.read_time()))
    gpflow.utilities.print_summary(m, fmt="notebook")

    if likelihood_func=="Gaussian" or model=="SGPR":
        final_hyperparams = {
            "kernel variance": m.kernel.variance.numpy(),
            "kernel lengthscale": m.kernel.lengthscales.numpy(),
            "likelihood variance": m.likelihood.variance.numpy(),
            "inducing locations": m.inducing_variable.Z.numpy(),
        }

        initial_hyperparams = {
            "kernel variance": initial_kernel_variance,
            "kernel lengthscale": initial_kernel_lengthscale,
            "initial noise variance": initial_likelihood_hyperparameter,
            "initial inducing locations": inducing_point_locations
        }
        
    elif likelihood=="RobustMax":
        final_hyperparams = {
            "kernel variance": m.kernel.variance.numpy(),
            "kernel lengthscale": m.kernel.lengthscales.numpy(),
            "likelihood invlink epsilon": m.likelihood.invlink.epsilon.numpy(),
            }
        initial_hyperparams = {
            "kernel variance": initial_kernel_variance,
            "kernel lengthscale": initial_kernel_lengthscale,
            #"initial noise variance": initial_likelihood_hyperparameter,
        }

    return (
        model,
        M,
        train_ind_points,
        learning_rate_adam,
        minibatch_size,
        num_iterations,
        likelihood_func,
        times,
        elbo_logs,
        accuracy_logs,
        initial_hyperparams,
        final_hyperparams,
        m
        )


#%% Dataframe of results

col_names =  [
    "Model",
    "M",
    "Train Z",
    "Adam learning rate",
    "minibatch size",
    "iterations",
    "likelihood",
    "time logs",
    "ELBO logs",
    "accuracy logs",
    "initial hyperparams",
    "final hyperparams",
    "GPflow model object"
    ]
  
# create an empty dataframe
# with columns
results_df  = pd.DataFrame(columns = col_names)
  
# show the dataframe
# results_df
# %%

results_df.loc[len(results_df.index)] = train_mnist("SGPR", 500, 200)

# %%

var_SVGP = results_df["final hyperparams"].loc[results_df.index[0]]["kernel variance"]
len_SVGP = results_df["final hyperparams"].loc[results_df.index[0]]["kernel lengthscale"]
noi_SVGP = results_df["final hyperparams"].loc[results_df.index[0]]["likelihood variance"]
Z_SVGP = results_df["final hyperparams"].loc[results_df.index[0]]["inducing locations"]

print(var_SVGP, len_SVGP, noi_SVGP)

results_df.loc[len(results_df.index)] = train_mnist(
    "SVGP",
    500,
    5000,
    "Random",
    False,
    "Gaussian",
    minibatch_size=500,
    initial_kernel_variance=var_SVGP,
    initial_kernel_lengthscale=len_SVGP, initial_likelihood_hyperparameter=noi_SVGP,
)

# %%

# plt.plot(results_df["time logs"].loc[results_df.index[0]], results_df["ELBO logs"].loc[results_df.index[0]])
# plt.plot(results_df["time logs"].loc[results_df.index[1]], results_df["ELBO logs"].loc[results_df.index[1]])
# plt.show()

# plt.plot(results_df["time logs"].loc[results_df.index[0]], results_df["accuracy logs"].loc[results_df.index[0]])
# plt.plot(results_df["time logs"].loc[results_df.index[1]], results_df["accuracy logs"].loc[results_df.index[1]])
# plt.grid(True, alpha=0.2)
# plt.show()
# %%
results_df.to_csv(
    "local_SparseGP_MNIST_results.csv"
    # mode="a",
    # header=False,
)
# %%

# m_sgpr = results_df["GPflow model object"].loc[results_df.index[0]]
# m_svgp = results_df["GPflow model object"].loc[results_df.index[1]]

# # %%
# mu, cov = m_sgpr.compute_qu()

# m_svgp
# # %%
# m_svgp.q_mu.assign(mu)
# m_svgp.q_sqrt.assign(cov)

# x_train, x_test, y_train, y_test = load_mnist(y_hot_encoding=False)

# res1=m_svgp.predict_y(x_test)

# res2=m_sgpr.predict_y(x_test)

# print(res1[:20].reshape(-1, 1))
# print(res2[:20].reshape(-1, 1))


# %%

# np.shape(cov)
# %%
