# ==========================================================
# Statistical Test Helpers
# ==========================================================
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
import pingouin as pg                       # pip install pingouin
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------- 1. Paired t-test --------------------------------
def paired_ttest(group1, group2, alpha=0.05):
    """
    Paired t-test for two related samples (e.g. same folds, two models).

    Parameters
    ----------
    group1, group2 : 1-D array-like, equal length
        Metric values for each paired observation (fold).
    alpha : float
        Significance threshold.

    Returns dict { 't', 'df', 'p', 'significant' }.
    """
    stat, p = ttest_rel(group1, group2)
    return {"t": stat, "df": len(group1)-1, "p": p, "significant": p < alpha}


# -------- 2. Wilcoxon signed-rank --------------------------
def wilcoxon_signed_rank(group1, group2, alpha=0.05):
    """
    Non-parametric alternative to the paired t-test.

    Returns dict { 'stat', 'p', 'significant' }.
    """
    stat, p = wilcoxon(group1, group2, zero_method="wilcox")
    return {"stat": stat, "p": p, "significant": p < alpha}


# -------- 3. Repeated-measures ANOVA -----------------------
def rm_anova(data, dv, within, subject):
    """
    One-way repeated-measures ANOVA via pingouin.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain metric column `dv`, factor column `within`,
        and subject identifier `subject`.
    Returns pingouin ANOVA table.
    """
    return pg.rm_anova(data=data, dv=dv, within=within, subject=subject)


# -------- 4. Friedman test (non-parametric RM-ANOVA) -------
def friedman_test(*groups):
    """
    Friedman test for k paired samples.
    Pass each group's metric vector as a separate argument.
    Returns (statistic, p_value).
    """
    return friedmanchisquare(*groups)


# -------- 5. Tukey HSD post-hoc ----------------------------
def tukey_hsd(values, labels, alpha=0.05):
    """
    Tukey’s Honest Significant Difference for all pairwise comparisons.

    Parameters
    ----------
    values : 1-D array-like
        Flattened metric values from all groups.
    labels : 1-D array-like
        Same length as `values`, giving the group name for each entry.

    Returns statsmodels TukeyHSDResults (call .summary()).
    """
    return pairwise_tukeyhsd(endog=values, groups=labels, alpha=alpha)


# ==========================================================
# Quick-start Examples  (commented out)
# ==========================================================

"""
# Example data: fold-wise F1 for three models
f1_A = np.array([0.81, 0.79, 0.83, 0.80, 0.82])
f1_B = np.array([0.78, 0.76, 0.79, 0.77, 0.78])
f1_C = np.array([0.75, 0.74, 0.76, 0.74, 0.75])

# 1) Paired t-test (A vs B)
print(paired_ttest(f1_A, f1_B))

# 2) Wilcoxon (A vs C, non-parametric)
print(wilcoxon_signed_rank(f1_A, f1_C))

# 3) Repeated-measures ANOVA (A,B,C)
df = pd.DataFrame({
    "fold":  np.tile(np.arange(1, 6), 3),
    "model": np.repeat(["A","B","C"], 5),
    "F1":    np.concatenate([f1_A, f1_B, f1_C])
})
print(rm_anova(df, dv="F1", within="model", subject="fold"))

# 4) Friedman (non-parametric omnibus)
print(friedman_test(f1_A, f1_B, f1_C))

# 5) Tukey HSD post-hoc
all_vals   = np.concatenate([f1_A, f1_B, f1_C])
all_labels = ["A"]*5 + ["B"]*5 + ["C"]*5
print(tukey_hsd(all_vals, all_labels).summary())
"""