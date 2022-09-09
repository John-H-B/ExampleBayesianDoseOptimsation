import numpy as np

"""
Data as gathered from:
Zhu et al. Safety, Tolerability, and Immunogenicity of a Recombinant Adenovirus Type-5Vectored COVID-19 Vaccine:
A Dose-Escalation, Open-Label, Non-Randomised, First-in-Human Trial. Lancet 2020, 395, 1845–1854.
"""

data_dose = np.log10([5 * 10 ** 10, 1 * 10 ** 11, 1.5 * 10 ** 11])
data_n = np.asarray([36, 36, 36])
data_eff = np.asarray([18, 18, 27])
data_tox = np.asarray([2, 2, 6])
