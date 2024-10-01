from scipy.special import gammaln
import numpy as np

def compute_likelihood(conf_matrix, total_samples, priors=None):
    """
    Computes the likelihood of a confusion matrix under a multinomial distribution.
    
    Parameters:
    conf_matrix (np.array): Confusion matrix of shape (n_classes,).
    total_samples (int): Total number of samples classified.
    priors (np.array): Prior probabilities for each class (optional).
    
    Returns:
    float: The likelihood of the observed data given the model (classifier).
    """
    if priors is None:
        # If no prior provided, assume uniform prior over classes
        priors = np.ones(len(conf_matrix)) / len(conf_matrix)
    
        # Compute the log-likelihood using gammaln for stability
    log_likelihood = (
        gammaln(total_samples + 1)  # log(total_samples!)
        - np.sum(gammaln(conf_matrix + 1))  # Sum of log(k_i!)
        + np.sum(conf_matrix * np.log(priors))  # Sum of k_i * log(p_i)
    )
    
    return log_likelihood


def bayesianHypothesisTesting(conf_matrix_1, conf_matrix_2):
    # Sum the rows to get the total predictions per class
    correct_1 = np.sum(conf_matrix_1, axis=0)
    correct_2 = np.sum(conf_matrix_2, axis=0)
    assert np.all(correct_1 != correct_2)

    total_samples = np.sum(correct_1)
    assert total_samples == np.sum(correct_2)

    # Compute likelihood for both classifiers assuming uniform prior
    log_likelihood_1 = compute_likelihood(correct_1, total_samples)
    log_likelihood_2 = compute_likelihood(correct_2, total_samples)

    # Compute the Bayes Factor
    BF_12 = np.exp(log_likelihood_1 - log_likelihood_2)

    # Compute posterior probabilities (assuming equal priors for both classifiers)
    posterior_M1 = BF_12 / (BF_12 + 1)
    posterior_M2 = 1 / (BF_12 + 1)

    # Print results
    print(f"Log Likelihood of Classifier 1: {log_likelihood_1}")
    print(f"Log Likelihood of Classifier 2: {log_likelihood_2}")
    print(f"Bayes Factor (Classifier 1 vs Classifier 2): {BF_12}")
    print(f"Posterior probability for Classifier 1: {posterior_M1}")
    print(f"Posterior probability for Classifier 2: {posterior_M2}")
