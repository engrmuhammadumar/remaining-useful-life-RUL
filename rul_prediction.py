"""
Implementation of the remaining useful lifetime (RUL) prediction framework proposed
by Nguyen et al. in 2025 for reinforced concrete beams monitored with acoustic
emission (AE) data.  The pipeline consists of four main stages:

1. **State indicator (SI) construction using the Mann–Whitney U test (MWUT)**:
   At each time step the raw AE signal is segmented and compared to a reference
   segment representing the normal condition.  A non‑parametric MWUT is used
   to measure how different the two samples are, and the resulting z‑score
   quantifies the instantaneous damage【419690026081095†screenshot】.

2. **Damage accumulation indicator (DA) construction**:  SIs are cumulatively
   summed to obtain DAs which represent the total accumulated damage up to a
   given time.  DAs are rescaled into the interval [0, 1] and new SIs are
   recomputed as differences between consecutive DAs.  These rescaled SIs serve
   as labels for supervised learning【419690026081095†screenshot】.

3. **DNN‑based SI construction**:  A stacked autoencoder (SAE) is pretrained on
   peak‑amplitude histograms derived from the AE signal to learn a compact
   representation.  The encoder is then fine‑tuned with a small neural network
   to map raw inputs to the rescaled SIs【419690026081095†screenshot】.  This
   process produces a more consistent SI across sensors and mitigates the
   variability inherent in raw MWUT calculations.

4. **RUL prognosis using a gated recurrent unit (GRU) network**:  Damage
   accumulation indicators constructed from the learned SIs are fed to a GRU
   model to predict future DA values.  The model is composed of an input
   layer of length 50 followed by two GRU layers of 100 units and a linear
   output layer【419690026081095†screenshot】.  A remaining useful lifetime is
   estimated by finding the time step at which the predicted DA crosses a
   predefined threshold (e.g., 0.9).

This script provides an end‑to‑end implementation of the above procedure using
the `numpy`, `scipy` and `tensorflow.keras` libraries.  It is designed to be
adaptable: the user only needs to change file paths and possibly the
segmentation length to suit their own AE dataset.  Note that `tensorflow`
is required for the deep models; install it via `pip install tensorflow` if
necessary.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.layers import (Input, Dense, Dropout, GRU, Lambda,
                                         Activation)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow is required for the DNN and GRU models. Please install"
        " it with 'pip install tensorflow'.") from exc


def load_ae_signals(data_dir: str) -> Dict[str, np.ndarray]:
    """Load raw AE signals from a directory.

    Each sensor is expected to have its own file (e.g. `sensor_1.npy` or
    `sensor_1.csv`).  The function reads all files with common numeric names
    and returns a dictionary mapping sensor identifiers to 1‑D numpy arrays.

    Parameters
    ----------
    data_dir: str
        Path to the directory containing AE data files.

    Returns
    -------
    Dict[str, np.ndarray]
        A mapping from sensor names (file names without extension) to the
        corresponding raw AE time series.
    """
    signals: Dict[str, np.ndarray] = {}
    for file_name in os.listdir(data_dir):
        name, ext = os.path.splitext(file_name)
        path = os.path.join(data_dir, file_name)
        if ext.lower() in {'.npy', '.npz'}:
            signals[name] = np.load(path)
        elif ext.lower() in {'.csv', '.txt'}:
            # assume single column of numbers
            signals[name] = np.loadtxt(path, delimiter=',')
    return signals


def segment_signal(signal: np.ndarray, segment_length: int) -> List[np.ndarray]:
    """Break a 1‑D signal into non‑overlapping segments of fixed length.

    Parameters
    ----------
    signal: np.ndarray
        The raw time series.
    segment_length: int
        Number of samples per segment (e.g. samples collected in one second).

    Returns
    -------
    List[np.ndarray]
        A list of segments.  The last segment is discarded if it is shorter
        than `segment_length`.
    """
    n_segments = len(signal) // segment_length
    return [signal[i * segment_length:(i + 1) * segment_length]
            for i in range(n_segments)]


def mann_whitney_z_score(x: np.ndarray, y: np.ndarray) -> float:
    """Compute the two‑sided Mann–Whitney U test z‑score between two samples.

    Rather than relying on external libraries, this function computes the
    U‑statistic and then standardises it according to its mean and variance.

    Parameters
    ----------
    x, y: np.ndarray
        Two 1‑D samples assumed to be independent.

    Returns
    -------
    float
        The standardised z‑score of the U statistic.  A larger absolute value
        indicates a greater difference between the two samples.
    """
    n_x = len(x)
    n_y = len(y)
    # Concatenate and compute ranks.  Rank ties are handled by average ranks.
    combined = np.concatenate((x, y))
    ranks = rankdata(combined)
    # Sum ranks for each sample
    rank_x = np.sum(ranks[:n_x])
    rank_y = np.sum(ranks[n_x:])
    # Compute U statistics for each sample
    u_x = rank_x - n_x * (n_x + 1) / 2
    u_y = rank_y - n_y * (n_y + 1) / 2
    u = min(u_x, u_y)
    # Mean and variance of U
    mean_u = n_x * n_y / 2.0
    var_u = n_x * n_y * (n_x + n_y + 1) / 12.0
    # Standard score (normal approximation)
    z = (u - mean_u) / np.sqrt(var_u + 1e-10)
    return z


def compute_mwut_si(segments: List[np.ndarray], reference_segments: List[np.ndarray]) -> np.ndarray:
    """Compute a series of MWUT‑based state indicators for a list of segments.

    Parameters
    ----------
    segments: List[np.ndarray]
        The segments for which SIs will be calculated.
    reference_segments: List[np.ndarray]
        Segments drawn from the normal condition.  A sliding window of these
        segments is randomly sampled for each comparison to mitigate noise.

    Returns
    -------
    np.ndarray
        An array of z‑scores (SIs) with the same length as `segments`.
    """
    s_len = len(reference_segments)
    si_values = []
    # Use the median reference segment to reduce computation
    # Alternatively, sample a subset for each calculation
    ref_sample = np.concatenate(reference_segments)
    for seg in segments:
        z = mann_whitney_z_score(seg, ref_sample)
        si_values.append(z)
    return np.array(si_values)


def construct_da(si: np.ndarray) -> np.ndarray:
    """Construct the damage accumulation indicator (DA) by cumulative sum.

    Parameters
    ----------
    si: np.ndarray
        Array of state indicators.

    Returns
    -------
    np.ndarray
        The cumulative sum of `si`.
    """
    return np.cumsum(si)


def rescale_da_and_compute_si(da: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rescale a DA to [0,1] and recompute the corresponding SI sequence.

    According to the paper, DAs from different sensors can have vastly different
    ranges.  To obtain consistent targets for the DNN, each DA is scaled to
    [0,1] and then differentiated to yield a new SI sequence【419690026081095†screenshot】.

    Parameters
    ----------
    da: np.ndarray
        The original damage accumulation indicator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the scaled DA and the recomputed SI sequence.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    da_scaled = scaler.fit_transform(da.reshape(-1, 1)).flatten()
    # SI(n) = DA(n) - DA(n-1); define SI(0) = DA(0)
    si_new = np.empty_like(da_scaled)
    si_new[0] = da_scaled[0]
    si_new[1:] = da_scaled[1:] - da_scaled[:-1]
    return da_scaled, si_new


def build_stacked_autoencoder(input_dim: int,
                              encoding_dims: List[int],
                              dropout_rate: float = 0.1) -> Tuple[Model, Model]:
    """Construct a stacked autoencoder for unsupervised pretraining.

    The architecture follows the description in the paper: the input is a
    2000‑dimensional vector representing a histogram of peak amplitudes; the
    encoder layers decrease in size (e.g. 1000 → 200 → 10) and are mirrored by
    decoder layers.  ELU activations and Xavier initialisation are used, and
    dropout is applied before each dense layer【419690026081095†screenshot】.

    Parameters
    ----------
    input_dim: int
        Dimensionality of the input vectors.
    encoding_dims: List[int]
        Sizes of the encoding layers, in decreasing order.
    dropout_rate: float
        Dropout rate applied before each dense layer.

    Returns
    -------
    Tuple[Model, Model]
        A tuple containing (autoencoder, encoder).  The encoder can later be
        reused as the feature extractor for supervised learning.
    """
    # Build encoder
    ae_input = Input(shape=(input_dim,), name='ae_input')
    x = ae_input
    # Keep track of intermediate layers for mirroring
    encoder_layers: List = []
    for dim in encoding_dims:
        x = Dropout(dropout_rate)(x)
        x = Dense(dim, activation='elu', kernel_initializer='glorot_uniform')(x)
        encoder_layers.append(x)
    encoded = encoder_layers[-1]
    # Build decoder (reverse of encoder)
    for dim in reversed(encoding_dims[:-1]):
        encoded = Dropout(dropout_rate)(encoded)
        encoded = Dense(dim, activation='elu', kernel_initializer='glorot_uniform')(encoded)
    decoded = Dropout(dropout_rate)(encoded)
    decoded = Dense(input_dim, activation='elu', kernel_initializer='glorot_uniform')(decoded)
    autoencoder = Model(ae_input, decoded, name='autoencoder')
    encoder = Model(ae_input, encoder_layers[-1], name='encoder')
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder, encoder


def build_dnn_from_encoder(encoder: Model,
                           dropout_rate: float = 0.2) -> Model:
    """Build a DNN for SI estimation on top of a pretrained encoder.

    The encoder layers are reused as the hidden layers, followed by a dropout
    layer and a final dense layer with a linear activation to predict the
    rescaled SI.  The network can be fine‑tuned end‑to‑end.

    Parameters
    ----------
    encoder: Model
        The pretrained encoder part of the SAE.
    dropout_rate: float
        Dropout rate applied before the output layer.

    Returns
    -------
    Model
        A compiled Keras model mapping input histograms to SI estimates.
    """
    encoder.trainable = True
    dnn_input = encoder.input
    x = encoder(dnn_input)
    x = Dropout(dropout_rate)(x)
    # Output layer with linear activation; SI values can be positive or negative
    output = Dense(1, activation='linear')(x)
    dnn_model = Model(dnn_input, output, name='si_constructor')
    dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return dnn_model


def prepare_peak_histogram(segment: np.ndarray, n_bins: int = 2000) -> np.ndarray:
    """Compute a peak‑amplitude histogram from a raw AE segment.

    The paper constructs histograms of peak amplitudes as inputs to the
    autoencoder and DNN【419690026081095†screenshot】.  Each segment is
    normalised and its absolute values are binned into a fixed number of bins.

    Parameters
    ----------
    segment: np.ndarray
        One AE segment.
    n_bins: int
        Number of histogram bins.

    Returns
    -------
    np.ndarray
        A 1‑D array of length `n_bins` representing the normalised histogram.
    """
    # Compute absolute peak values within the segment
    abs_vals = np.abs(segment)
    # Normalise by the maximum to stabilise bin edges
    if abs_vals.max() > 0:
        abs_vals = abs_vals / abs_vals.max()
    # Histogram
    hist, _ = np.histogram(abs_vals, bins=n_bins, range=(0.0, 1.0), density=False)
    # Normalise histogram to sum to 1
    hist = hist.astype('float32')
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def build_gru_prognosis_model(input_length: int = 50,
                              n_units: int = 100) -> Model:
    """Construct the GRU‑based RUL prognosis model.

    The architecture follows the description in the paper: an input sequence
    length of 50 time steps, two GRU layers of 100 units each, and a dense
    output layer with linear activation【419690026081095†screenshot】.

    Parameters
    ----------
    input_length: int
        Number of historical time steps used for predicting the next DA value.
    n_units: int
        Number of GRU units in each recurrent layer.

    Returns
    -------
    Model
        A compiled Keras model that accepts input of shape (batch_size,
        input_length, 1) and outputs a single future value.
    """
    inp = Input(shape=(input_length, 1), name='gru_input')
    x = GRU(n_units, return_sequences=True)(inp)
    x = GRU(n_units)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out, name='gru_prognosis')
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model


def create_sequences(data: np.ndarray, window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a 1‑D time series into supervised learning sequences.

    Each sample consists of `window` consecutive values as input and the
    following value as the target.  Samples are created with a step size of 1.

    Parameters
    ----------
    data: np.ndarray
        Input time series (e.g. damage accumulation indicator).
    window: int
        Number of past points to use for each prediction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A pair `(X, y)` where `X` has shape (n_samples, window, 1) and `y`
        has shape (n_samples, 1).
    """
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def predict_da_trajectory(model: Model,
                          initial_window: np.ndarray,
                          steps: int) -> List[float]:
    """Iteratively predict future DA values using the trained GRU model.

    Parameters
    ----------
    model: Model
        A trained GRU prognosis model.
    initial_window: np.ndarray
        The last `window` ground‑truth DA values before prediction begins.
        Shape should be (window,).
    steps: int
        Number of future steps to predict.

    Returns
    -------
    List[float]
        A list containing the predicted DA values (length `steps`).
    """
    seq = initial_window.astype('float32').copy()
    preds: List[float] = []
    window = len(initial_window)
    for _ in range(steps):
        inp = seq[-window:].reshape(1, window, 1)
        pred = model.predict(inp, verbose=0).flatten()[0]
        preds.append(pred)
        seq = np.append(seq, pred)
    return preds


def estimate_rul_from_da(da_true: np.ndarray,
                         da_pred: np.ndarray,
                         threshold: float = 0.9) -> Tuple[int, int]:
    """Estimate remaining useful life from predicted DA trajectories.

    The failure time is defined as the first index where the DA crosses
    `threshold`.  The RUL is simply the difference between the predicted failure
    index and the current index at which prediction began.  The function
    returns both the true failure index and the predicted failure index.

    Parameters
    ----------
    da_true: np.ndarray
        Ground‑truth DA sequence.
    da_pred: np.ndarray
        Predicted DA sequence, concatenated with the ground‑truth history so
        that its indices align with `da_true`.
    threshold: float
        Failure threshold on DA (default 0.9, as used in the paper).

    Returns
    -------
    Tuple[int, int]
        A tuple `(true_fail_idx, pred_fail_idx)`.  If the threshold is never
        crossed, the corresponding value is `-1`.
    """
    true_fail_idx = int(np.argmax(da_true >= threshold))
    if da_true[true_fail_idx] < threshold:
        true_fail_idx = -1
    pred_fail_idx = int(np.argmax(da_pred >= threshold))
    if da_pred[pred_fail_idx] < threshold:
        pred_fail_idx = -1
    return true_fail_idx, pred_fail_idx


def example_usage():  # pragma: no cover
    """Example of how to use the functions in this module.

    This function demonstrates the end‑to‑end workflow using synthetic data.
    Replace the synthetic generation with actual AE data loading and
    segmentation when applying to the real dataset.
    """
    # ---------------------------------------------------------------------
    # 1. Load raw AE signals
    # ---------------------------------------------------------------------
    # data_dir = '/path/to/ae_dataset'
    # signals = load_ae_signals(data_dir)
    # For demonstration we generate a synthetic signal with a gradual change
    np.random.seed(42)
    synthetic_signal = np.random.randn(100000) * 0.1
    synthetic_signal[50000:] += np.linspace(0, 5, 50000)
    # ---------------------------------------------------------------------
    # 2. Segment the signal and compute MWUT‑based SI
    # ---------------------------------------------------------------------
    segment_len = 1000  # adjust according to sampling frequency (samples per second)
    segments = segment_signal(synthetic_signal, segment_len)
    # Use the first 10 % of segments as reference normal condition
    n_ref = max(1, len(segments) // 10)
    reference_segments = segments[:n_ref]
    mwut_si = compute_mwut_si(segments, reference_segments)
    # ---------------------------------------------------------------------
    # 3. Construct DA and rescale it; compute new SI labels
    # ---------------------------------------------------------------------
    da = construct_da(mwut_si)
    da_scaled, si_labels = rescale_da_and_compute_si(da)
    # ---------------------------------------------------------------------
    # 4. Compute peak‑amplitude histograms for each segment
    # ---------------------------------------------------------------------
    histograms = np.array([prepare_peak_histogram(seg) for seg in segments])
    # ---------------------------------------------------------------------
    # 5. Pretrain stacked autoencoder and fine‑tune DNN
    # ---------------------------------------------------------------------
    input_dim = histograms.shape[1]
    encoding_dims = [1000, 200, 10]
    autoencoder, encoder = build_stacked_autoencoder(input_dim, encoding_dims)
    autoencoder.fit(histograms, histograms,
                    epochs=20, batch_size=32, verbose=0)
    dnn_model = build_dnn_from_encoder(encoder)
    dnn_model.fit(histograms, si_labels,
                  epochs=20, batch_size=32, verbose=0)
    # ---------------------------------------------------------------------
    # 6. Use the DNN to construct new SIs and DAs
    # ---------------------------------------------------------------------
    si_constructed = dnn_model.predict(histograms, verbose=0).flatten()
    da_constructed = construct_da(si_constructed)
    da_con_scaled, _ = rescale_da_and_compute_si(da_constructed)
    # ---------------------------------------------------------------------
    # 7. Prepare sequences and train the GRU prognosis model
    # ---------------------------------------------------------------------
    window = 50
    X_train, y_train = create_sequences(da_scaled, window)
    X_test, y_test = create_sequences(da_con_scaled, window)
    gru_model = build_gru_prognosis_model(window, 100)
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
    gru_model.fit(X_train, y_train,
                  epochs=50, batch_size=64,
                  validation_split=0.1,
                  callbacks=callbacks,
                  verbose=0)
    # ---------------------------------------------------------------------
    # 8. Predict future DA trajectory and estimate RUL
    # ---------------------------------------------------------------------
    # Choose a starting point (e.g. 300 steps before the end)
    start_idx = len(da_con_scaled) - 300
    history = da_con_scaled[start_idx - window:start_idx]
    future_steps = 300
    predicted_future = predict_da_trajectory(gru_model, history, future_steps)
    predicted_full = np.concatenate([da_con_scaled[:start_idx], predicted_future])
    true_full = da_scaled[:len(predicted_full)]
    true_fail_idx, pred_fail_idx = estimate_rul_from_da(true_full, predicted_full)
    print(f"True failure index: {true_fail_idx}")
    print(f"Predicted failure index: {pred_fail_idx}")


if __name__ == '__main__':  # pragma: no cover
    # The example usage is provided for illustration purposes and is not
    # executed when this module is imported.  Remove the leading underscore
    # to run the example with synthetic data.
    # example_usage()
    pass