# Copyright (c) 2021, Kamilė Stankevičiūtė
# Adapted from Ahmed M. Alaa github.com/ahmedmalaa/rnn-blockwise-jackknife
# Licensed under the BSD 3-clause license

# ---------------------------------------------------------
# Helper functions and utilities for deep learning models
# ---------------------------------------------------------

import numpy as np
import torch

from models.bjrnn import RNN_uncertainty_wrapper
from models.dprnn import DPRNN
from models.qrnn import QRNN


def evaluate_cfrnn_performance(model, test_dataset, correct_conformal=True):
    # Todo: add quantile loss as a metric for evaluation
    lower_quantile_loss, upper_quantile_loss= model.evaluate_quantile_loss(test_dataset, corrected=correct_conformal)
    print(f'lower_quantile_loss in performance.py: {lower_quantile_loss}')
    print(f'upper_quantile_loss in performance.py: {upper_quantile_loss}')



    independent_coverages, joint_coverages, intervals = model.evaluate_coverage(
        test_dataset, corrected=correct_conformal
    )
    mean_independent_coverage = torch.mean(independent_coverages.float(), dim=0)
    mean_joint_coverage = torch.mean(joint_coverages.float(), dim=0).item()
    interval_widths = (intervals[:, 1] - intervals[:, 0]).squeeze()
    point_predictions, errors = model.get_point_predictions_and_errors(test_dataset, corrected=correct_conformal)

    results = {
        "Point predictions": point_predictions,
        "Errors": errors,
        "Independent coverage indicators": independent_coverages.squeeze(),
        "Joint coverage indicators": joint_coverages.squeeze(),
        "Upper limit": intervals[:, 1],
        "Lower limit": intervals[:, 0],
        "Mean independent coverage": mean_independent_coverage.squeeze(),
        "Mean joint coverage": mean_joint_coverage,
        "Confidence interval widths": interval_widths,
        "Mean confidence interval widths": interval_widths.mean(dim=0),
    }

    return results


def evaluate_performance(model, X_test, Y_test, coverage=0.9):
    print(type(model))
    if type(model) is RNN_uncertainty_wrapper:
        y_pred, y_l_approx, y_u_approx = model.predict(X_test, coverage=coverage)

    elif type(model) is QRNN:
        # evaluate quantile loss in test dataset
        lower_quantile_loss, upper_quantile_loss= model.evaluate_quantile_loss(X_test,Y_test,coverage)
        print(f'lower_quantile_loss in performance.py: {lower_quantile_loss}')
        print(f'upper_quantile_loss in performance.py: {upper_quantile_loss}')

        # y_u_approx, y_l_approx = model.predict(X_test) is wrong, since the first element is the lower bound and the second element is the upper bound
        y_l_approx, y_u_approx = model.predict(X_test)
        # print(f'y_u_approx in performance.py: {y_u_approx[0]}')
        # print(f'y_l_approx in performance.py: {y_l_approx[0]}')
        y_pred = [(y_l_approx[k] + y_u_approx[k]) / 2 for k in range(len(y_u_approx))]

        y_pred = [x.reshape(-1, 1) for x in y_pred]
        y_u_approx = [x.reshape(-1, 1) for x in y_u_approx]
        y_l_approx = [x.reshape(-1, 1) for x in y_l_approx]

    elif type(model) is DPRNN:
        y_pred, y_std = model.predict(X_test, alpha=1 - coverage)
        y_u_approx = [y_pred[k] + y_std[k] for k in range(len(y_pred))]
        y_l_approx = [y_pred[k] - y_std[k] for k in range(len(y_pred))]

        y_pred = [x.reshape(-1, 1) for x in y_pred]
        y_u_approx = [x.reshape(-1, 1) for x in y_u_approx]
        y_l_approx = [x.reshape(-1, 1) for x in y_l_approx]

    results = {}

    upper = np.array(y_u_approx).squeeze()
    lower = np.array(y_l_approx).squeeze()
    pred = np.array(y_pred).squeeze()
    target = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in Y_test]).squeeze()

    results["Point predictions"] = np.array(y_pred)
    results["Upper limit"] = np.array(y_u_approx)
    results["Lower limit"] = np.array(y_l_approx)
    results["Confidence interval widths"] = upper - lower
    results["Mean confidence interval widths"] = np.mean(upper - lower, axis=0)
    results["Errors"] = np.abs(target - pred)

    independent_coverage = np.logical_and(upper >= target, lower <= target)

    results["Independent coverage indicators"] = independent_coverage
    results["Joint coverage indicators"] = np.all(independent_coverage, axis=1)
    results["Mean independent coverage"] = np.mean(independent_coverage, axis=0)
    results["Mean joint coverage"] = np.mean(results["Joint coverage indicators"])

    return results
