"""Hybrid factor risk model (spec:
docs/superpowers/specs/2026-09-22-factor-risk-model-design.md).

Macro factors are fixed ETF portfolios, residualized level by level, with
time-series betas projected from one long-span EWM covariance ``S``:
``F = W'^T S W'``, ``B = S W' F^-1``. Momentum is characteristic-based: the
exposure is the standardized 12-1 month return, and the factor return is a
cross-sectional regression of macro residual returns on it. Idiosyncratic risk
is floored at ``residual_floor * S_ii``. ``B`` and ``D`` are then rescaled by
``sigma_short / sqrt(S_ii)``, so short-span vols ride on long-span
correlations. The factor side carries its own short-span vol: ``F`` is scaled
by ``g g^T`` and ``B`` divided by ``g`` (``g`` = short-span / long-span factor
vol), which leaves ``Sigma`` unchanged but keeps exposures in real units, the
units of the realized factor returns. Everything is in daily units, and
streaming/causal: the snapshot after ``update(bar t)`` uses data up to and
including bar t only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from backtester.core.events import Ticker

from trading_strategies.risk.config import FactorModelConfig
from trading_strategies.risk.factors import factor_weight_matrix, load_factors, residualize
from trading_strategies.risk.style_factors import MomentumSignal, momentum_exposures
from trading_strategies.utils.streaming import EwmCovariance, EwmMoments, FloatArray

MOMENTUM = "Momentum"
IDIOSYNCRATIC = "Idiosyncratic"
_EIGEN_FLOOR_RELATIVE = 1e-8
_EIGEN_FLOOR_ABSOLUTE = 1e-18
_VARIANCE_EPS = 1e-18


@dataclass(frozen=True)
class RiskModelSnapshot:
    tickers: tuple[Ticker, ...]
    factors: tuple[str, ...]
    B: FloatArray  # N x K, vol-rescaled betas / exposures
    F: FloatArray  # K x K factor covariance
    D: FloatArray  # N specific variances
    sigma: FloatArray  # N short-span daily vols
    factor_weights: FloatArray  # N x K_macro residualized factor portfolios
    macro_betas: FloatArray  # N x K_macro long-span betas (for macro residuals)
    momentum: FloatArray | None  # N momentum exposures, even before Momentum joins F
    ready: bool

    @classmethod
    def empty(cls) -> RiskModelSnapshot:
        matrix = np.zeros((0, 0))
        vector = np.zeros(0)
        return cls((), (), matrix, matrix, vector, vector, matrix, matrix, None, False)

    def covariance(self) -> FloatArray:
        return self.B @ self.F @ self.B.T + np.diag(self.D)

    def exposures(self, w: FloatArray) -> FloatArray:
        return self.B.T @ w

    def portfolio_variance(self, w: FloatArray) -> float:
        x = self.exposures(w)
        return float(x @ self.F @ x + np.sum(self.D * w**2))

    def risk_decomposition(self, w: FloatArray) -> dict[str, float]:
        variance = self.portfolio_variance(w)
        if variance <= 0:
            return dict.fromkeys((*self.factors, IDIOSYNCRATIC), 0.0)
        x = self.exposures(w)
        marginal = self.F @ x
        shares = {name: float(x[k] * marginal[k]) / variance for k, name in enumerate(self.factors)}
        shares[IDIOSYNCRATIC] = float(np.sum(self.D * w**2)) / variance
        return shares

    def factor_returns(self, asset_returns: Mapping[Ticker, float]) -> FloatArray:
        r = self._return_vector(asset_returns)
        macro = self.factor_weights.T @ r
        if MOMENTUM not in self.factors:
            return macro
        return np.append(macro, self._momentum_return(r, macro))

    def momentum_return(self, asset_returns: Mapping[Ticker, float]) -> float | None:
        if self.momentum is None:
            return None
        r = self._return_vector(asset_returns)
        return self._momentum_return(r, self.factor_weights.T @ r)

    def _return_vector(self, asset_returns: Mapping[Ticker, float]) -> FloatArray:
        r = np.array([asset_returns.get(t, 0.0) for t in self.tickers], dtype=np.float64)
        return np.nan_to_num(r)

    def _momentum_return(self, r: FloatArray, macro: FloatArray) -> float:
        assert self.momentum is not None
        # OLS residual (see _regress): orthogonal to the macro factor
        # returns by construction, Cov(macro_residual, macro) == 0. This is
        # what gives momentum its orthogonality to the macro factors, with
        # no explicit residualize() call needed.
        macro_residual = r - self.macro_betas @ macro
        denom = float(self.momentum @ self.momentum)
        return float(self.momentum @ macro_residual) / denom if denom > 0 else 0.0


class FactorRiskModel:
    def __init__(self, tickers: Sequence[Ticker], config: FactorModelConfig) -> None:
        self._tickers = tuple(tickers)
        self._known = frozenset(self._tickers)
        self._config = config
        self._definitions = load_factors(Path(config.factors_file), granular=config.granular)
        # Two independent covariance estimates over the same returns, at
        # different spans. _corr (long span) gives the correlation structure
        # used for residualization and betas: needs 250 days to be stable,
        # short-span correlations would make the factor definitions jump
        # around bar to bar. _vol (short span) gives the asset vols (sigma):
        # needs to react fast to a vol regime change, a 250-day vol lags a
        # spike for months. `scale = sigma / sqrt(s_ii)` below then rescales
        # the long-span-based B/D into short-span-vol units, so short-span
        # vols ride on long-span correlations.
        self._vol = EwmCovariance(self._tickers, span=config.vol_span, min_periods=config.vol_span)
        self._corr = EwmCovariance(
            self._tickers, span=config.corr_span, min_periods=config.corr_min_periods
        )
        self._momentum = MomentumSignal(
            self._tickers, lookback=config.momentum.lookback, skip=config.momentum.skip
        )
        # Long-span (corr_span) factor variance: F needs to sit on the same
        # span as the macro correlations so the whole covariance is estimated
        # consistently. Short-span (vol_span) twin: only used to derive
        # g_momentum below, the vol-ratio rescale that lets the momentum
        # factor react to the current vol regime the way asset sigma does.
        # min_periods=vol_span on both: momentum only starts producing a
        # factor return once macro residuals exist, so there is no reason to
        # make it wait for the full long span to warm up before either one
        # starts accumulating.
        self._momentum_var = EwmMoments(span=config.corr_span, min_periods=config.vol_span)
        self._momentum_short_var = EwmMoments(span=config.vol_span, min_periods=config.vol_span)
        self._last_close: dict[Ticker, float] = {}
        self._snapshot = RiskModelSnapshot.empty()

    @property
    def snapshot(self) -> RiskModelSnapshot:
        return self._snapshot

    def update(self, closes: Mapping[Ticker, float]) -> None:
        live = {t: c for t, c in closes.items() if t in self._known and math.isfinite(c) and c > 0}
        returns = {
            t: math.log(c / self._last_close[t]) for t, c in live.items() if t in self._last_close
        }
        self._last_close.update(live)
        # Previous snapshot + this bar's returns: causal.
        momentum_return = self._snapshot.momentum_return(returns)
        if momentum_return is not None:
            self._momentum_var.update(momentum_return)
            self._momentum_short_var.update(momentum_return)
        self._vol.update(returns)
        self._corr.update(returns)
        self._momentum.update(live)
        self._snapshot = self._build()

    def _build(self) -> RiskModelSnapshot:
        corr = self._corr.covariance()
        vol = self._vol.covariance()
        universe = _complete_universe(corr, vol)
        if len(universe) < 2:
            return RiskModelSnapshot.empty()
        tickers = tuple(self._tickers[i] for i in universe)
        s = corr[np.ix_(universe, universe)]
        v = vol[np.ix_(universe, universe)]
        sigma = np.sqrt(np.diag(vol)[universe])

        macro = self._macro_block(tickers, s, v)
        if macro is None:
            return RiskModelSnapshot.empty()
        factor_weights, macro_betas, f_macro, names, g = macro
        b, f, g, names, momentum = self._append_momentum(
            tickers, sigma, macro_betas, f_macro, names, g
        )
        return self._finalize(
            tickers, s, sigma, b, f, g, factor_weights, macro_betas, names, momentum
        )

    def _macro_block(
        self, tickers: tuple[Ticker, ...], s: FloatArray, v: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, list[str], FloatArray] | None:
        """Residualized macro factor portfolios (``factor_weights``), their
        betas (``macro_betas``), factor covariance (``f_macro``) and vol
        ratio (``g``). ``None`` if no macro factor is active for this ticker
        universe."""
        raw_weights = factor_weight_matrix(self._definitions, tickers, available=set(tickers))
        active = np.flatnonzero(np.any(raw_weights != 0, axis=0))
        if active.size == 0:
            return None
        definitions = [self._definitions[k] for k in active]
        factor_weights = residualize(raw_weights[:, active], [d.level for d in definitions], s)
        f_macro = _clip_psd(factor_weights.T @ s @ factor_weights)
        macro_betas = _regress(s, factor_weights, f_macro)
        names = [d.name for d in definitions]
        g = _vol_ratio(np.diag(factor_weights.T @ v @ factor_weights), np.diag(f_macro))
        return factor_weights, macro_betas, f_macro, names, g

    def _append_momentum(
        self,
        tickers: tuple[Ticker, ...],
        sigma: FloatArray,
        macro_betas: FloatArray,
        f_macro: FloatArray,
        names: list[str],
        g: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, list[str], FloatArray | None]:
        """Appends the momentum factor to ``(B, F, g, names)`` once its
        variance is warm. Momentum needs no explicit ``residualize()`` step
        to be orthogonal to the macro factors: its factor return is a
        cross-sectional combination of macro *residual* returns (see
        ``RiskModelSnapshot._momentum_return``), and those residuals are
        already orthogonal to the macro factor returns because
        ``macro_betas`` is the OLS projection computed by ``_regress``."""
        momentum = self._momentum_exposure(tickers, sigma)
        momentum_std = self._momentum_var.std
        if momentum is None or not momentum_std:
            return macro_betas, f_macro, g, names, momentum
        k = len(names)
        b = np.column_stack([macro_betas, momentum])
        f = np.block(
            [
                [f_macro, np.zeros((k, 1))],
                [np.zeros((1, k)), np.array([[momentum_std**2]])],
            ]
        )
        names = [*names, MOMENTUM]
        short_std = self._momentum_short_var.std
        # Same trick as the macro g above: short-span vol / long-span vol
        # for the momentum factor. Rescaling F's momentum block and B's
        # momentum column by this ratio (in _finalize) leaves Sigma unchanged
        # but puts the momentum exposure in units of the factor's current,
        # not stale long-span, vol.
        g_momentum = (
            _vol_ratio(np.array([short_std**2]), np.array([momentum_std**2]))
            if short_std is not None
            else np.ones(1)
        )
        return b, f, np.append(g, g_momentum), names, momentum

    def _finalize(
        self,
        tickers: tuple[Ticker, ...],
        s: FloatArray,
        sigma: FloatArray,
        b: FloatArray,
        f: FloatArray,
        g: FloatArray,
        factor_weights: FloatArray,
        macro_betas: FloatArray,
        names: list[str],
        momentum: FloatArray | None,
    ) -> RiskModelSnapshot:
        """Floors specific risk, rescales ``B``/``F`` from long-span to
        short-span vol units (short-span vols riding on long-span
        correlations, see the module docstring), and packages the immutable
        ``RiskModelSnapshot``."""
        s_ii = np.diag(s)
        common = np.einsum("ik,kl,il->i", b, f, b)
        specific = np.maximum(self._config.residual_floor * s_ii, s_ii - common)
        scale = sigma / np.sqrt(s_ii)
        # Sigma = diag(scale) (b f b^T + D) diag(scale), split so that F has
        # short-span factor vols and B stays in real units.
        arrays = [
            b * scale[:, None] / g[None, :],
            f * np.outer(g, g),
            specific * scale**2,
            sigma,
            factor_weights,
            macro_betas,
        ]
        if momentum is not None:
            arrays.append(momentum)
        for array in arrays:
            array.flags.writeable = False
        return RiskModelSnapshot(
            tickers=tickers,
            factors=tuple(names),
            B=arrays[0],
            F=arrays[1],
            D=arrays[2],
            sigma=sigma,
            factor_weights=factor_weights,
            macro_betas=macro_betas,
            momentum=momentum,
            ready=True,
        )

    def _momentum_exposure(
        self, tickers: tuple[Ticker, ...], sigma: FloatArray
    ) -> FloatArray | None:
        raw = {t: value for t in tickers if (value := self._momentum.raw_return(t)) is not None}
        exposures = momentum_exposures(raw, dict(zip(tickers, sigma.tolist(), strict=True)))
        if not exposures:
            return None
        return np.array([exposures.get(t, 0.0) for t in tickers])


def _complete_universe(corr: FloatArray, vol: FloatArray) -> list[int]:
    """Tickers with warm vol and correlation, and a finite covariance with
    every other such ticker."""
    candidates = [
        i
        for i in range(corr.shape[0])
        if np.isfinite(corr[i, i]) and corr[i, i] > 0 and np.isfinite(vol[i, i]) and vol[i, i] > 0
    ]
    block = corr[np.ix_(candidates, candidates)]
    return [c for j, c in enumerate(candidates) if np.isfinite(block[j]).all()]


def _regress(cov: FloatArray, weights: FloatArray, factor_cov: FloatArray) -> FloatArray:
    """OLS regression coefficient of returns on the factor returns
    ``f = weights.T @ r``, under the covariance ``cov``:
    ``Cov(r, f) @ inv(Cov(f, f))``. The residual ``r - result @ f`` is
    therefore orthogonal to ``f`` by construction (the defining property of
    a least-squares projection) -- this is what gives the momentum factor
    return its orthogonality to the macro factors
    (see ``RiskModelSnapshot._momentum_return``), with no explicit
    ``residualize()`` call needed."""
    return cov @ weights @ np.linalg.pinv(factor_cov)


def _vol_ratio(short_variance: FloatArray, long_variance: FloatArray) -> FloatArray:
    """Short-span over long-span vol, per factor. 1.0 where the short-span
    variance is not finite (a pair not yet warm in the short window)."""
    ratio = np.sqrt(np.maximum(short_variance, _VARIANCE_EPS)) / np.sqrt(
        np.maximum(long_variance, _VARIANCE_EPS)
    )
    return np.where(np.isfinite(ratio), ratio, 1.0)


def _clip_psd(matrix: FloatArray) -> FloatArray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    floor = max(_EIGEN_FLOOR_RELATIVE * float(values.max()), _EIGEN_FLOOR_ABSOLUTE)
    return (vectors * np.maximum(values, floor)) @ vectors.T
