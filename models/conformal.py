import numpy as np
import torch
from torch import nn


def nonconformity(output, target):
    """Measures the nonconformity between output and target time series."""
    # Average MAE loss for every step in the sequence.
    return torch.nn.functional.l1_loss(output, target, reduction='none')


def cover(pred, target):
    # Returns True when the entire forecast fits into predicted conformal
    # intervals.
    return torch.all(
        torch.logical_and(target >= pred[:, 0], target <= pred[:, 1])).item()


class ConformalForecaster(nn.Module):
    def __init__(self, embedding_size, input_size=1, horizon=1,
                 error_rate=0.05):
        super(ConformalForecaster, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.

        # Encoder and forecaster can be the same (if embeddings are
        # trained on `horizon`-step forecasts), but different models are
        # possible.

        # TODO try separate encoder and forecaster models.
        # TODO try the RNN autoencoder trained on reconstruction error.
        self.encoder = None

        # Single-shot multi-output univariate time series forecaster.
        # https://www.tensorflow.org/tutorials/structured_data/time_series#rnn_2
        # TODO consider autoregressive multi-output model:
        # https://www.tensorflow.org/tutorials/structured_data/time_series#advanced_autoregressive_model
        self.forecaster_rnn = nn.LSTM(input_size=input_size,
                                      hidden_size=embedding_size,
                                      batch_first=True)
        self.forecaster_out = nn.Linear(embedding_size, horizon)

        self.num_train = None
        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.alpha = error_rate

    def forward(self, x):
        _, (h_n, c_n) = self.forecaster_rnn(x)
        out = self.forecaster_out(h_n)

        return out

    def fit(self, dataset, calibration_dataset, epochs, lr, batch_size=150):
        # Train the forecaster to return correct multi-step predictions.
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        self.num_train = len(dataset)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            train_loss = 0.

            for sequences, targets in train_loader:  # iterate through batches
                optimizer.zero_grad()

                out = self(sequences)

                loss = criterion(out, targets)
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            print('Epoch: {}\tTrain loss: {}'.format(epoch, mean_train_loss))

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def calibrate(self, calibration_dataset):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                                                         batch_size=1)
        calibration_scores = []

        with torch.set_grad_enabled(False):
            self.eval()
            for sequences, targets in calibration_loader:
                out = self(sequences)
                calibration_scores.append(nonconformity(out, targets))

        self.calibration_scores = calibration_scores

        # Given p_{z}:=\frac{\left|\left\{i=m+1, \ldots, n+1: R_{i} \geq R_{n+1}\right\}\right|}{n-m+1}
        # and the accepted R_{n+1} = \Delta(y, f(x_{test})) are such that
        # p_{z} > \alpha we have that the nonconformity scores should be below
        # the (corrected) (1 - alpha)% of calibration scores.
        self.critical_calibration_scores = torch.tensor([np.quantile(
            position_calibration_scores, q=1 - self.alpha * self.num_train / (
                    self.num_train + 1))
            for position_calibration_scores in self.calibration_scores])

    def predict(self, x):
        """Forecasts the time series with conformal uncertainty intervals."""
        out = self(x)
        # TODO +/- nonconformity will not return *adaptive* interval widths.
        return torch.vstack([out - self.critical_calibration_scores,
                             out + self.critical_calibration_scores]).T
