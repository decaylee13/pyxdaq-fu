"""
learning.py

Learning trajectory analysis for autoresearch_training/.

This module is what fundamentally distinguishes autoresearch_training/ from
autoresearch/: instead of evaluating point-in-time performance, it measures
whether performance is improving over time — i.e., whether the culture is
genuinely learning from the stimulation feedback signal.

Key concept: drift correction
    Biological cultures change spontaneously over time regardless of stimulation
    (changes in firing rates, network state, etc.).  This "biological drift"
    can masquerade as learning (or anti-learning).  subtract_drift() removes
    the drift component by subtracting the slope observed under a control
    (no-stim) condition from the slope observed under the active regime.
    Only the drift-corrected delta should be used for keep/discard decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from autoresearch_training.session_runner import SessionResult


@dataclass
class LearningCurve:
    regime_id: str
    session_results: list[SessionResult]
    slope: float                  # linear regression slope of mean_rally_length over sessions
    slope_pvalue: float           # p-value of the slope (two-tailed, H0: slope == 0)
    baseline_mean: float          # mean of first 2 sessions (early performance)
    late_mean: float              # mean of last 2 sessions (late performance)
    delta: float                  # late_mean - baseline_mean
    is_learning: bool             # True if slope > 0 and p < learning_pvalue_threshold
    is_declining: bool            # True if slope < 0 and p < learning_pvalue_threshold
    n_sessions: int


def compute_learning_curve(
    results: list[SessionResult],
    learning_pvalue_threshold: float = 0.10,
) -> LearningCurve:
    """
    Compute a LearningCurve from a list of SessionResults.

    Uses linear regression of mean_rally_length over session_index to estimate
    the learning slope.  Crashed sessions are excluded from the fit; if all
    sessions crashed, returns a flat curve with crashed=True semantics.

    Parameters
    ----------
    results                   : Sessions in the order they were run.
    learning_pvalue_threshold : p-value cutoff for declaring learning significant.
                                Default 0.10 is lenient, appropriate given small N.
    """
    if not results:
        raise ValueError("Cannot compute LearningCurve from an empty list")

    regime_id = results[0].regime_id
    valid     = [r for r in results if not r.crashed]

    if len(valid) < 2:
        # Degenerate: not enough data for regression
        mean_val = valid[0].mean_rally_length if valid else 0.0
        return LearningCurve(
            regime_id=regime_id,
            session_results=results,
            slope=0.0,
            slope_pvalue=1.0,
            baseline_mean=mean_val,
            late_mean=mean_val,
            delta=0.0,
            is_learning=False,
            is_declining=False,
            n_sessions=len(valid),
        )

    xs = [float(r.session_index) for r in valid]
    ys = [r.mean_rally_length    for r in valid]

    slope, intercept, r_value, p_value, std_err = _linregress(xs, ys)

    n = len(valid)
    baseline_mean = sum(r.mean_rally_length for r in valid[:2]) / 2
    late_mean     = sum(r.mean_rally_length for r in valid[-2:]) / max(len(valid[-2:]), 1)
    delta         = late_mean - baseline_mean

    return LearningCurve(
        regime_id=regime_id,
        session_results=results,
        slope=slope,
        slope_pvalue=p_value,
        baseline_mean=baseline_mean,
        late_mean=late_mean,
        delta=delta,
        is_learning=(slope > 0 and p_value < learning_pvalue_threshold),
        is_declining=(slope < 0 and p_value < learning_pvalue_threshold),
        n_sessions=n,
    )


def estimate_sessions_needed(
    current_curve: LearningCurve,
    target_delta: float = 1.0,
) -> int:
    """
    Estimate how many more sessions are needed to reach target_delta improvement,
    based on current slope.

    Returns -1 if slope is flat or negative (learning not progressing).
    """
    if current_curve.slope <= 0:
        return -1

    sessions_to_target = target_delta / current_curve.slope
    sessions_run       = current_curve.n_sessions
    remaining          = int(math.ceil(max(0.0, sessions_to_target - sessions_run)))
    return remaining


def subtract_drift(
    regime_curve: LearningCurve,
    control_curve: LearningCurve,
) -> LearningCurve:
    """
    Correct for biological drift by subtracting the control condition slope.

    Biological drift is the spontaneous change in neural performance that occurs
    regardless of stimulation (e.g., culture maturation, electrode fouling, or
    network state shifts over hours).  Running periodic no-stim control sessions
    provides an empirical estimate of this drift.

    Correction method:
        drift_corrected_slope = regime_slope - control_slope
        drift_corrected_delta = regime_delta - control_slope * n_sessions_regime

    The returned LearningCurve represents the component of improvement (or decline)
    that is attributable to the stimulation feedback specifically, with the
    spontaneous drift component subtracted out.

    This is the primary method for isolating genuine learning from biological drift.
    It must be called before any keep/discard decision — never compare raw curves.

    Parameters
    ----------
    regime_curve  : LearningCurve from sessions under the active stimulation regime.
    control_curve : LearningCurve from no-stim control sessions (same time window).

    Returns
    -------
    A new LearningCurve with:
        slope  = regime_curve.slope - control_curve.slope
        delta  = corrected estimate of stimulation-driven improvement
        p-value recomputed using t-test on the slope difference (see implementation)
    """
    corrected_slope = regime_curve.slope - control_curve.slope

    # Corrected delta: how much of the observed delta exceeds what drift alone predicts
    # over the same session-index span.
    #
    # We use the x-coordinate span (last_x - baseline_x) rather than n_sessions, because
    # delta is measured from the mean of the first 2 sessions to the mean of the last 2.
    # Using n_sessions would overestimate drift by ~1 session depending on session indices.
    valid_r = [r for r in regime_curve.session_results if not r.crashed]
    baseline_x = sum(r.session_index for r in valid_r[:2]) / max(len(valid_r[:2]), 1)
    late_x     = sum(r.session_index for r in valid_r[-2:]) / max(len(valid_r[-2:]), 1)
    x_span     = late_x - baseline_x

    expected_drift  = control_curve.slope * x_span
    corrected_delta = regime_curve.delta - expected_drift

    # Recompute p-value for the corrected slope via t-test on slope difference.
    # With small N, we propagate uncertainty by combining standard errors.
    # se_diff = sqrt(se_regime^2 + se_control^2) — conservative (independent estimates).
    se_regime  = _slope_se(regime_curve)
    se_control = _slope_se(control_curve)
    se_diff    = math.sqrt(se_regime**2 + se_control**2)

    if se_diff == 0:
        p_corrected = 0.0 if corrected_slope != 0 else 1.0
    else:
        t_stat      = corrected_slope / se_diff
        df          = max(regime_curve.n_sessions + control_curve.n_sessions - 4, 1)
        p_corrected = _t_pvalue(t_stat, df)

    # The corrected baseline/late means are identical to the raw regime values —
    # we only adjust the slope/delta estimates, not the raw session observations.
    return LearningCurve(
        regime_id=regime_curve.regime_id,
        session_results=regime_curve.session_results,
        slope=corrected_slope,
        slope_pvalue=p_corrected,
        baseline_mean=regime_curve.baseline_mean,
        late_mean=regime_curve.late_mean,
        delta=corrected_delta,
        is_learning=(corrected_slope > 0 and p_corrected < 0.10),
        is_declining=(corrected_slope < 0 and p_corrected < 0.10),
        n_sessions=regime_curve.n_sessions,
    )


def detect_saturation(curve: LearningCurve, window: int = 3) -> bool:
    """
    Return True if improvement has plateaued over the last `window` sessions.

    Plateau is defined as: the slope over the last `window` sessions is not
    significantly positive (i.e., improvement has flattened or reversed).
    """
    valid = [r for r in curve.session_results if not r.crashed]
    if len(valid) < window:
        return False

    recent = valid[-window:]
    xs = [float(r.session_index) for r in recent]
    ys = [r.mean_rally_length    for r in recent]

    if len(set(xs)) < 2:
        return True  # all same index — degenerate

    slope, _, _, _, _ = _linregress(xs, ys)
    return slope <= 0.0


# ---------------------------------------------------------------------------
# Statistical helpers (stdlib only — no numpy/scipy required)
# ---------------------------------------------------------------------------

def _linregress(
    xs: list[float], ys: list[float]
) -> tuple[float, float, float, float, float]:
    """
    Simple linear regression. Returns (slope, intercept, r_value, p_value, std_err).
    Attempts to use scipy.stats.linregress; falls back to a pure-stdlib implementation.
    """
    try:
        from scipy.stats import linregress as _scipy_lr
        result = _scipy_lr(xs, ys)
        return float(result.slope), float(result.intercept), float(result.rvalue), float(result.pvalue), float(result.stderr)
    except ImportError:
        pass

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    if ss_xx == 0:
        return 0.0, mean_y, 0.0, 1.0, 0.0

    slope     = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    r_value = ss_xy / math.sqrt(ss_xx * ss_yy) if ss_yy > 0 else 0.0

    residuals  = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res     = sum(r ** 2 for r in residuals)
    df         = n - 2
    std_err    = math.sqrt(ss_res / df / ss_xx) if df > 0 and ss_xx > 0 else 0.0

    t_stat     = slope / std_err if std_err > 0 else 0.0
    p_value    = _t_pvalue(t_stat, df) if df > 0 else 1.0

    return slope, intercept, r_value, p_value, std_err


def _slope_se(curve: LearningCurve) -> float:
    """Extract the standard error of the slope from a LearningCurve."""
    valid = [r for r in curve.session_results if not r.crashed]
    if len(valid) < 3:
        return 1.0   # high uncertainty with very few points

    xs = [float(r.session_index) for r in valid]
    ys = [r.mean_rally_length    for r in valid]
    _, _, _, _, std_err = _linregress(xs, ys)
    return max(std_err, 1e-9)


def _t_pvalue(t: float, df: int) -> float:
    """
    Two-tailed p-value for a t-statistic with `df` degrees of freedom.
    Uses the regularised incomplete beta function via math.lgamma (stdlib).
    """
    try:
        from scipy.stats import t as _scipy_t
        return float(_scipy_t.sf(abs(t), df) * 2)
    except ImportError:
        pass

    # Pure stdlib approximation via betainc
    x = df / (df + t * t)
    p_one_tail = _betainc(df / 2.0, 0.5, x) / 2.0
    return min(1.0, max(0.0, 2.0 * p_one_tail))


def _betainc(a: float, b: float, x: float) -> float:
    """
    Regularised incomplete beta function I_x(a, b) approximated via continued fraction.
    Sufficient accuracy for the small t-values expected here.
    """
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a

    # Lentz continued fraction
    def _cf(a: float, b: float, x: float) -> float:
        eps, tiny = 1e-12, 1e-30
        f = tiny
        C, D = f, 0.0
        for m in range(200):
            for j in (0, 1):
                if m == 0 and j == 0:
                    d = 1.0
                elif j == 0:
                    d = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                else:
                    d = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                D = 1.0 + d * D
                if abs(D) < tiny:
                    D = tiny
                D = 1.0 / D
                C = 1.0 + d / C
                if abs(C) < tiny:
                    C = tiny
                delta = C * D
                f *= delta
                if abs(delta - 1.0) < eps:
                    return f
        return f

    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x)
    else:
        lbeta2 = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        front2 = math.exp(math.log(1 - x) * b + math.log(x) * a - lbeta2) / b
        return 1.0 - front2 * _cf(b, a, 1 - x)
