import math
import numpy as np
import math
import scipy.stats

def paired_t_test_statistic(population_1, population_2):
    """
    Computes the test statistic of a paired t-test
    
    Inputs:
        - population_1, np.array 1D: an array of 1 dimension with the observations of 1st group
        - population_2, np.array 1D: an array of 1 dimension with the observations of 2nd group
    
    Output: the test statistic t
    """
    dim1,dim2=population_1.shape[0],population_2.shape[0]
    assert(dim1 == dim2), f"populations have different size, got: dim1={dim1}, dim2={dim2}."
    mean_diff = np.mean(population_1 - population_2)
    s_diff = np.std(population_1 - population_2)
    standard_errors = s_diff / math.sqrt(len(population_1))

    return mean_diff / standard_errors


def paired_t_test(population_1, population_2, quantile=.05, type="TWO"):
    """
    Computes the result of the paired t-test given 2 groups of observations and a level of significance
    """

    statistic = paired_t_test_statistic(population_1, population_2)
    if type == "LEFT":
        "alternative shows pop_1<pop_2"
        p_value = scipy.stats.t.sf(-statistic, df=len(population_1) - 1)
    elif type == "RIGHT":
        "alternative shows pop_1>pop_2"
        p_value = scipy.stats.t.sf(statistic, df=len(population_1) - 1)
    elif type == "TWO":
        "alternative shows pop_1!=pop_2"
        p_value = scipy.stats.t.sf(abs(statistic), df=len(population_1) - 1) * 2
    else:
        raise Exception("Incorrect type! The type has to be one of the following: ['UPPER', 'LOWER', 'EQUAL']")
    # critical_value = paired_t_test_critical_value(threshold, len(population_1)-1, type)
    if p_value > quantile:
        return False, p_value
    else:
        return True, p_value

def equivalence_t_test(population_1, population_2, threshold=0.1, quantile=0.05):
    """
    Computes the equivalence test to show that mean 2 - threshold < mean 1 < mean 2 + threshold

    Inputs:
        - population_1, np.array 1D: an array of 1 dimension with the observations of 1st group
        - population_2, np.array 1D: an array of 1 dimension with the observations of 2nd group
        - threshold, float: small value that determines the maximal difference that can be between means
        - quantile, float: the level of significance, is the 1-q quantile of the distribution
    
    Output, bool: True if the null is rejected and False if there is not enough evidence
    """
    population_2_up = population_2 + threshold
    population_2_low = population_2 - threshold
    if paired_t_test(population_1, population_2_low, quantile, type="RIGHT") and\
    paired_t_test(population_1, population_2_up, quantile, type="LEFT"):

        print("null hypothesis rejected at a level of significance", quantile)
        return True, max(paired_t_test(population_1, population_2_low, quantile, type="RIGHT")[1],
                        paired_t_test(population_1, population_2_up, quantile, type="LEFT")[1])
    else:
        print("not enough evidence to reject null hypothesis")
        return False, max(paired_t_test(population_1, population_2_low, quantile, type="RIGHT")[1],
                        paired_t_test(population_1, population_2_up, quantile, type="LEFT")[1])
