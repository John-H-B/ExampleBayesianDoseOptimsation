"""
An example of Bayesian optimisation.
"""

import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from data import data_dose, data_n, data_eff, data_tox
from functions import *
from setup import Setup

_s = Setup()

domain = np.linspace(_s.domain_min, _s.domain_max, _s.domain_resolution)

"""
Efficacy
"""

# Plot data and maximum likelihood fit
nll_eff = lambda *args: -log_likelihood_eff(*args)
initial = np.array([1, 7.5, 1])
MLE_eff = minimize(nll_eff, initial, method='Nelder-Mead', args=(data_dose, [data_n, data_eff])).x
pred_max_eff, pred_mid_eff, pred_scale_eff = MLE_eff

plt.scatter(data_dose, data_eff / data_n)
plt.plot(domain, efficacy_model(domain, pred_max_eff, pred_mid_eff, pred_scale_eff))
plt.title(f'MLE dose-efficacy curve \n'
          f'Max: {pred_max_eff:.3f}, Mid: {pred_mid_eff:.3f}, Scale: {pred_scale_eff:.3f} ')
plt.xlabel('log10 Dose')
plt.xlabel('Probability of efficacy')
plt.savefig('figures/MLE_efficacy.png')
plt.close()

## Markov-Chain Monte-Carlo
initial = MLE_eff + 1e-4 * np.random.randn(_s.mcmc_chains, 3)
number_chains, num_params = initial.shape

sampler = emcee.EnsembleSampler(
    number_chains, num_params, log_probability_eff, args=(data_dose, [data_n, data_eff])
)
sampler.run_mcmc(initial, _s.mcmc_samples, progress=True)

if _s.show_mcmc_convergence:
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["max", "mid", "scale"]
    for i in range(num_params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.suptitle('Convergence plot for dose-efficacy model')
    plt.savefig('figures/convergence_efficacy.png')
    plt.close()

# Posterior Efficacy Plots
posterior_eff_sample_parameters = sampler.get_chain(discard=100, thin=15, flat=True)

array = np.zeros((_s.posterior_samples, _s.domain_resolution))

inds = np.random.randint(len(posterior_eff_sample_parameters), size=_s.posterior_samples)
for ind in inds:
    sample = posterior_eff_sample_parameters[ind]
    plt.plot(domain, efficacy_model(domain, sample[0], sample[1], sample[2]), "C1", alpha=0.1)
plt.scatter(data_dose, data_eff / data_n, label='Data')
plt.legend(fontsize=14)
plt.xlim(0, 15)
plt.title('Posterior Predictions of Dose Efficacy')
plt.xlabel('log10 Dose')
plt.ylabel("Predicted percentage efficacy")
plt.legend()
plt.savefig('figures/posterior_efficacy.png')
plt.close()

for idx, ind in enumerate(inds):
    sample = posterior_eff_sample_parameters[ind]
    pred = efficacy_model(domain, sample[0], sample[1], sample[2])
    array[idx, :] = pred

plt.scatter(data_dose, data_eff / data_n, label='Data')
for i in np.linspace(0.1, 0.9, 9):
    plt.plot(domain, np.quantile(array, i, 0), color='blue', alpha=0.3)
plt.plot(domain, np.quantile(array, 0.5, 0), color='black', label='Median Prediction')
plt.plot(domain, efficacy_model(domain, pred_max_eff, pred_mid_eff, pred_scale_eff), label='MLE Prediction')
plt.title('Posterior 10% Interval Predictions of Dose Efficacy')
plt.xlabel('log10 Dose')
plt.ylabel("Predicted percentage efficacy")
plt.legend()
plt.savefig('figures/posterior_intervals_efficacy.png')
plt.close()

"""
Toxicity
"""

# Plot data and maximum likelihood fit
nll_tox = lambda *args: -log_likelihood_tox(*args)
initial = np.array([7.5, 1])
MLE_tox = minimize(nll_tox, initial, method='Nelder-Mead', args=(data_dose, [data_n, data_tox])).x
pred_mid_tox, pred_scale_tox = MLE_tox

plt.scatter(data_dose, data_tox / data_n)
plt.plot(domain, toxicity_model(domain, pred_mid_tox, pred_scale_tox))
plt.title(f'MLE dose-toxicity curve \n'
          f'Mid: {pred_mid_tox:.3f}, Scale: {pred_scale_tox:.3f} ')
plt.xlabel('log10 Dose')
plt.xlabel('Probability of toxicity')
plt.savefig('figures/MLE_toxicity.png')
plt.close()

## Markov-Chain Monte-Carlo

initial = MLE_tox + 1e-4 * np.random.randn(_s.mcmc_chains, 2)
number_chains, num_params = initial.shape

sampler = emcee.EnsembleSampler(
    number_chains, num_params, log_probability_tox, args=(data_dose, [data_n, data_tox])
)
sampler.run_mcmc(initial, _s.mcmc_samples, progress=True)

if _s.show_mcmc_convergence:
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["mid", "scale"]
    for i in range(num_params):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")
    fig.suptitle('Convergence plot for dose-toxicity model')
    plt.savefig('figures/convergence_toxicity.png')
    plt.close()

# Posterior toxicity Plots
posterior_tox_sample_parameters = sampler.get_chain(discard=100, thin=15, flat=True)

array = np.zeros((_s.posterior_samples, _s.domain_resolution))

inds = np.random.randint(len(posterior_tox_sample_parameters), size=_s.posterior_samples)
for ind in inds:
    sample = posterior_tox_sample_parameters[ind]
    plt.plot(domain, toxicity_model(domain, sample[0], sample[1]), "C1", alpha=0.1)
plt.xlim(0, 15)
plt.title('Posterior Predictions of Dose Toxicity')
plt.xlabel("Dose")
plt.ylabel("Predicted percentage Toxicity")
plt.savefig('figures/posterior_toxicity.png')
plt.close()

for idx, ind in enumerate(inds):
    sample = posterior_tox_sample_parameters[ind]
    pred = toxicity_model(domain, sample[0], sample[1])
    array[idx, :] = pred

plt.scatter(data_dose, data_tox / data_n, label='Data')
for i in np.linspace(0.1, 0.9, 9):
    plt.plot(domain, np.quantile(array, i, 0), color='blue', alpha=0.3)
plt.plot(domain, np.quantile(array, 0.5, 0), color='black', label='Median Prediction')
plt.plot(domain, toxicity_model(domain, pred_mid_tox, pred_scale_tox), label='MLE Prediction')
plt.title('Posterior 10% Interval Predictions of Dose toxicity')
plt.xlabel('log10 Dose')
plt.ylabel("Predicted percentage toxicity")
plt.legend()
plt.savefig('figures/posterior_intervals_toxicity.png')
plt.close()

"""
Utility
"""

array = np.zeros((_s.posterior_samples, _s.domain_resolution))
record_of_predicted_optimal_dose = []

# Sampling Utility
for idx, ind in enumerate(inds):
    sample_eff = posterior_eff_sample_parameters[ind]
    sample_tox = posterior_tox_sample_parameters[ind]
    pred_eff = efficacy_model(domain, sample_eff[0], sample_eff[1], sample_eff[2])
    pred_tox = toxicity_model(domain, sample_tox[0], sample_tox[1])
    pred_utility = utility(pred_eff, pred_tox)

    argmax = np.argmax(pred_utility)
    record_of_predicted_optimal_dose.append(domain[argmax])

    array[idx, :] = pred_utility

# Utility Intervals
for i in np.linspace(0.1, 0.9, 9):
    plt.plot(domain, np.quantile(array, i, 0), color='green', alpha=0.3)
plt.plot(domain, np.quantile(array, 0.5, 0), color='black', label='Median Prediction')
plt.title('Posterior 10% Interval Predictions of Dose-Utility')
plt.xlabel('log10 Dose')
plt.ylabel("Utility")

solutions = []

argmax = np.argmax(np.quantile(array, 0.1, 0))
best_dose = domain[argmax]
solutions.append(best_dose)
plt.scatter(best_dose, np.quantile(array[:, argmax], 0.1, 0), c='red', label='Pessimistic')
plt.vlines(best_dose, 0, np.quantile(array[:, argmax], 0.1, 0), colors='red')

argmax = np.argmax(np.quantile(array, 0.5, 0))
best_dose = domain[argmax]
solutions.append(best_dose)
plt.scatter(best_dose, np.quantile(array[:, argmax], 0.5, 0), c='black', label='Realistic')
plt.vlines(best_dose, 0, np.quantile(array[:, argmax], 0.5, 0), colors='black')

argmax = np.argmax(np.quantile(array, 0.9, 0))
best_dose = domain[argmax]
solutions.append(best_dose)
plt.scatter(best_dose, np.quantile(array[:, argmax], 0.9, 0), c='purple', label='Optimistic')
plt.vlines(best_dose, 0, np.quantile(array[:, argmax], 0.9, 0), colors='purple')

plt.xlim(8, 13)
plt.hlines(0, 8, 13)
plt.legend()
plt.savefig('figures/posterior_intervals_utility.png')
plt.close()

print(f'Best Dose if pessimistic: {(10 ** solutions[0]) / 10 ** 11:.2f} x 10^11 VP')
print(f'Best Dose if realistic: {(10 ** solutions[1]) / 10 ** 11:.2f} x 10^11 VP')
print(f'Best Dose if optimistic: {(10 ** solutions[2]) / 10 ** 11:.2f} x 10^11 VP')

print(np.quantile(record_of_predicted_optimal_dose, 0.1))
print(np.quantile(record_of_predicted_optimal_dose, 0.5))
print(np.quantile(record_of_predicted_optimal_dose, 0.9))

print(f'Median of credible optimal doses: '
      f'{10 ** np.quantile(record_of_predicted_optimal_dose, 0.5) / 10 ** 11:.2f} x 10^11 VP')
print(f'90% Credible interval for optimal dose: ['
      f'{10 ** np.quantile(record_of_predicted_optimal_dose, 0.05) / 10 ** 11:.2f} , '
      f'{10 ** np.quantile(record_of_predicted_optimal_dose, 0.95) / 10 ** 11:.2f}]'
      f' x 10^11 VP')

plt.hist(record_of_predicted_optimal_dose)
plt.title('Approximate probability histogram of optimal dose')
plt.xlabel('log10 Dose')
plt.savefig('figures/posterior_probability_of_optimal_dose.png')
plt.close()
