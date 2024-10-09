from scipy.special import gammaln
import numpy as np

def log_likelihood(confusion_matrix):
    # Ensure the input is a numpy array and has shape (7, 7)
    if not isinstance(confusion_matrix, np.ndarray):
        raise ValueError("Confusion matrix must be a numpy array")
    if confusion_matrix.shape != (7, 7):
        raise ValueError("Confusion matrix must have shape (7, 7)")

    # Total number of samples (sum of all entries in the confusion matrix)
    total_samples = np.sum(confusion_matrix)
    
    # To prevent issues with log(0), we add a small value (epsilon)
    epsilon = 1e-10

    # Calculate the likelihood for each class
    likelihood = 0
    for i in range(7):
        # Sum of counts for the true class `i`
        true_class_count = np.sum(confusion_matrix[i, :])
        
        # Prevent division by zero if there are no true instances of a class
        if true_class_count == 0:
            continue
        
        # Calculate the probability of predicting class i correctly
        p_correct = confusion_matrix[i, i] / true_class_count
        
        # Add the log-likelihood, avoiding log(0) by adding epsilon
        likelihood += np.log(p_correct + epsilon)

    return likelihood


def bayesianHypothesisTesting(conf_matrix_1, conf_matrix_2):
    total_samples = np.sum(conf_matrix_1)
    assert total_samples == np.sum(conf_matrix_2), "Confusion matrices must have the same number of samples"

    # Compute likelihood for both classifiers assuming uniform prior
    log_likelihood_1 = log_likelihood(conf_matrix_1, total_samples)
    log_likelihood_2 = log_likelihood(conf_matrix_2, total_samples)

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


def dirichlet_log_Z(u):
    """Calculate the log of normalization constant Z(u) for a Dirichlet prior."""
    return np.sum(gammaln(u)) - gammaln(np.sum(u))

def bayesian_hypothesis_testing(confusion_matrix_a, confusion_matrix_b, prior_correct=1, prior_error=0.01):
    """
    Bayesian Hypothesis Testing between two classifiers with modified counts and priors.
    
    Parameters:
    - confusion_matrix_a: Confusion matrix of classifier A (7x7 matrix).
    - confusion_matrix_b: Confusion matrix of classifier B (7x7 matrix).
    - prior_correct: Prior value for the diagonal elements (correct classifications).
    - prior_error: Prior value for off-diagonal elements (errors).
    
    Returns:
    - Bayes factor indicating the strength of evidence for the classifiers being different.
    """
    
    # Extract diagonal (correct classifications) and errors for each confusion matrix
    diagonal_a = np.diag(confusion_matrix_a)
    diagonal_b = np.diag(confusion_matrix_b)
    
    errors_a = np.sum(confusion_matrix_a, axis=1) - diagonal_a
    errors_b = np.sum(confusion_matrix_b, axis=1) - diagonal_b

    # Create count vectors according to the structure [c1, c2, ..., c7, e1, e2, ..., e7]
    ca = np.concatenate((diagonal_a, errors_a))
    cb = np.concatenate((diagonal_b, errors_b))

    # Create prior vector: uniform for correct classifications, near-zero for errors
    u = np.concatenate((np.ones(7) * prior_correct, np.ones(7) * prior_error))

    # Calculate Z(u)
    log_Z_u = dirichlet_log_Z(u)
    log_Z_u_ca = dirichlet_log_Z(u + ca)
    log_Z_u_cb = dirichlet_log_Z(u + cb)
    log_Z_u_ca_cb = dirichlet_log_Z(u + ca + cb)

    # Bayes factor calculation
    log_bayes_factor = log_Z_u_ca + log_Z_u_cb - log_Z_u - log_Z_u_ca_cb
    bayes_factor = np.exp(log_bayes_factor)

    # Print the result
    if bayes_factor > 1:
        print(f"Strong evidence in favor of classifiers being different (Bayes factor = {bayes_factor:.3f})")
    elif bayes_factor < 1:
        print(f"Strong evidence in favor of classifiers being the same (Bayes factor = {bayes_factor:.3f})")
    else:
        print(f"Neutral evidence (Bayes factor = {bayes_factor:.3f})")