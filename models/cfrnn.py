# Copyright (c) 2021, NeurIPS 2021 Paper6977 Authors
# Licensed under the BSD 3-clause license

import torch


def coverage(intervals, target):
    """ Determines whether intervals coverage the target prediction.

    Depending on the coverage_mode (either 'joint' or 'independent), will return
    either a list of whether each target or all targets satisfy the coverage.

    intervals: shape [batch_size, 2, horizon, n_outputs]
    """

    lower, upper = intervals[:, 0], intervals[:, 1]

    # [batch, horizon, n_outputs]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)

    # [batch, horizon, n_outputs], [batch, n_outputs]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)


def get_critical_scores(calibration_scores, q):
    return torch.tensor([[torch.quantile(
        position_calibration_scores,
        q=q)
        for position_calibration_scores in feature_calibration_scores]
        for feature_calibration_scores in calibration_scores]).T


class CFRNN(torch.nn.Module):
    def __init__(self, embedding_size, input_size=1, output_size=1, horizon=1,
                 error_rate=0.05, rnn_mode='LSTM', **kwargs):
        super(CFRNN, self).__init__()
        # input_size indicates the number of features in the time series
        # input_size=1 for univariate series.
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.horizon = horizon
        self.output_size = output_size
        self.alpha = error_rate

        self.rnn_mode = rnn_mode
        if self.rnn_mode == 'RNN':
            self.forecaster_rnn = torch.nn.RNN(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        elif self.rnn_mode == 'GRU':
            self.forecaster_rnn = torch.nn.GRU(input_size=input_size,
                                               hidden_size=embedding_size,
                                               batch_first=True)
        else:  # self.mode == 'LSTM'
            self.forecaster_rnn = torch.nn.LSTM(input_size=input_size,
                                                hidden_size=embedding_size,
                                                batch_first=True)
        self.forecaster_out = torch.nn.Linear(embedding_size,
                                              horizon * output_size)

        self.calibration_scores = None
        self.critical_calibration_scores = None
        self.corrected_critical_calibration_scores = None

    def forward(self, x, state=None):
        if state is not None:
            h_0, c_0 = state
        else:
            h_0 = None

        # [batch, horizon, output_size]
        if self.rnn_mode == "LSTM":
            _, (h_n, c_n) = self.forecaster_rnn(x.float(), state)
        else:
            _, h_n = self.forecaster_rnn(x.float(), h_0)
            c_n = None

        out = self.forecaster_out(h_n).reshape(-1, self.horizon,
                                               self.output_size)

        return out, (h_n, c_n)

    def get_lengths_mask(self, sequences, lengths):
        """Returns the lengths mask indicating the positions where every
        sequences in the batch are valid."""

        lengths_mask = torch.zeros(sequences.size(0), self.horizon,
                                   sequences.size(2))
        for i, l in enumerate(lengths):
            lengths_mask[i, :min(l, self.horizon), :] = 1

        return lengths_mask

    def train_forecaster(self, train_loader, epochs, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.train()
        for epoch in range(epochs):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                out, _ = self(sequences)
                valid_out = out * self.get_lengths_mask(sequences, lengths)

                loss = criterion(valid_out.float(), targets.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 50 == 0:
                print(
                    'Epoch: {}\tTrain loss: {}'.format(epoch, mean_train_loss))

    def nonconformity(self, output, calibration_example):
        """Measures the nonconformity between output and target time series."""
        # Average MAE loss for every step in the sequence.
        target = calibration_example[1]
        return torch.nn.functional.l1_loss(output, target, reduction='none')

    def calibrate(self, calibration_dataset):
        """
        Computes the nonconformity scores for the calibration dataset.
        """
        calibration_loader = torch.utils.data.DataLoader(calibration_dataset,
                                                         batch_size=1)
        n_calibration = len(calibration_dataset)
        calibration_scores = []

        with torch.set_grad_enabled(False):
            self.eval()
            for calibration_example in calibration_loader:
                sequences, targets, lengths = calibration_example
                out, _ = self(sequences)
                score = self.nonconformity(out, calibration_example)
                # n_batches: [batch_size, horizon, output_size]
                calibration_scores.append(score)

        # [output_size, horizon, n_samples]
        self.calibration_scores = torch.vstack(calibration_scores).T

        # [horizon, output_size]
        q = min((n_calibration + 1.) * (1 - self.alpha) / n_calibration, 1)
        corrected_q = min((n_calibration + 1.) * (
                1 - self.alpha / self.horizon) / n_calibration, 1)

        self.critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores,
            q=q)

        # Bonferroni corrected calibration scores.
        # [horizon, output_size]
        self.corrected_critical_calibration_scores = get_critical_scores(
            calibration_scores=self.calibration_scores,
            q=corrected_q)

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            batch_size=32):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Train the multi-horizon forecaster.
        self.train_forecaster(train_loader, epochs, lr)

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self(x, state)

        if not corrected:
            # [batch_size, horizon, n_outputs]
            lower = out - self.critical_calibration_scores
            upper = out + self.critical_calibration_scores
        else:
            # [batch_size, horizon, n_outputs]
            lower = out - self.corrected_critical_calibration_scores
            upper = out + self.corrected_critical_calibration_scores

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden

    def evaluate_coverage(self, test_dataset, corrected=True):
        self.eval()

        independent_coverages, joint_coverages, intervals = [], [], []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            batch_intervals, _ = self.predict(sequences, corrected=corrected)
            intervals.append(batch_intervals)
            independent_coverage, joint_coverage = coverage(batch_intervals,
                                                            targets)
            independent_coverages.append(independent_coverage)
            joint_coverages.append(joint_coverage)

        # [n_samples, (1 | horizon), n_outputs] containing booleans
        independent_coverages = torch.cat(independent_coverages)
        joint_coverages = torch.cat(joint_coverages)

        # [n_samples, 2, horizon, n_outputs] containing lower and upper bounds
        intervals = torch.cat(intervals)

        return independent_coverages, joint_coverages, intervals

    def get_point_predictions_and_errors(self, test_dataset, corrected=True,
                                         normalised=False):
        self.eval()

        point_predictions = []
        errors = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        for sequences, targets, lengths in test_loader:
            point_prediction, _ = self(sequences)
            batch_intervals, _ = self.predict(sequences, corrected=corrected,
                                              normalised=normalised)
            point_predictions.append(point_prediction)
            errors.append(torch.nn.functional.l1_loss(point_prediction,
                                                      targets,
                                                      reduction='none').squeeze())

        point_predictions = torch.cat(point_predictions)
        errors = torch.cat(errors)

        return point_predictions, errors


class CFRNN_normalised(CFRNN):
    def __init__(self, beta=1, **kwargs):
        super(CFRNN, self).__init__()

        # Normalisation network
        self.normalising_rnn = torch.nn.RNN(input_size=self.input_size,
                                            hidden_size=self.embedding_size,
                                            batch_first=True)

        self.normalising_out = torch.nn.Linear(self.embedding_size,
                                               self.horizon * self.output_size)

        self.beta = beta

        self.normalised_calibration_scores = None
        self.normalised_critical_calibration_scores = None
        self.normalised_corrected_critical_calibration_scores = None

    def normaliser_forward(self, sequences):
        """Returns an estimate of normalisation target ln|y - hat{y}|."""
        _, h_n = self.normalising_rnn(sequences.float())
        out = self.normalising_out(h_n).reshape(-1, self.horizon,
                                                self.output_size)
        return out

    def normalisation_score(self, sequences, lengths):
        out = self.normaliser_forward(sequences)
        return torch.exp(out) + self.beta

    def train_normaliser(self, train_loader):
        # TODO tuning
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        # TODO early stopping based on validation loss of the calibration set
        for epoch in range(500):
            train_loss = 0.

            for sequences, targets, lengths in train_loader:
                optimizer.zero_grad()

                # Get the RNN multi-horizon forecast.
                forecaster_out, _ = self(sequences)
                lengths_mask = self.get_lengths_mask(sequences, lengths)

                # Compute normalisation target ln|y - \hat{y}|.
                normalisation_target = \
                    torch.log(torch.abs(targets - forecaster_out)) * \
                    lengths_mask

                # Normalising network estimates the normalisation target.
                out = self.normaliser_forward(sequences)

                loss = criterion(out.float(), normalisation_target.float())
                loss.backward()

                train_loss += loss.item()

                optimizer.step()

            mean_train_loss = train_loss / len(train_loader)
            if epoch % 100 == 0:
                print(
                    'Epoch: {}\tNormalisation loss: {}'.format(epoch,
                                                               mean_train_loss))

    def nonconformity(self, output, calibration_example):
        """Measures the nonconformity between output and target time series."""
        sequence, target, length = calibration_example
        score = torch.nn.functional.l1_loss(output, target, reduction='none')
        normalised_score = score / self.normalisation_score(sequence, length)
        return normalised_score

    def fit(self, train_dataset, calibration_dataset, epochs, lr,
            batch_size=32):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

        # Train the multi-horizon forecaster.
        self.train_forecaster(train_loader, epochs, lr)

        # Train normalisation network.
        self.train_normaliser(train_loader)
        self.normalising_rnn.eval()
        self.normalising_out.eval()

        # Collect calibration scores
        self.calibrate(calibration_dataset)

    def predict(self, x, state=None, corrected=True, normalised=False):
        """Forecasts the time series with conformal uncertainty intervals."""
        out, hidden = self(x, state)

        score = self.normalisation_score(x, len(x))
        if not corrected:
            # [batch_size, horizon, n_outputs]
            # TODO make sure len(x) is correct
            lower = out - self.critical_calibration_scores * score
            upper = out + self.critical_calibration_scores * score

        else:
            # [batch_size, horizon, n_outputs]
            lower = out - \
                    self.corrected_critical_calibration_scores * score
            upper = out + \
                    self.corrected_critical_calibration_scores * score

        # [batch_size, 2, horizon, n_outputs]
        return torch.stack((lower, upper), dim=1), hidden