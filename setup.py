class Setup():
    def __init__(self):
        self.domain_min = 0.0
        self.domain_max = 15.0
        self.domain_resolution = 101

        self.eff_max_prior_alpha = 6.0
        self.eff_max_prior_beta = 1.0
        self.eff_mid_prior_mu = 8.0
        self.eff_mid_prior_sigma = 4.0

        self.tox_mid_prior_mu = 8.0
        self.tox_mid_prior_sigma = 4.0

        self.mcmc_samples = 1000
        self.mcmc_chains = 32
        self.burn = 200
        self.posterior_samples = 500

        self.show_mcmc_convergence = True

        assert self.mcmc_samples > self.posterior_samples
