# Factor definitions (`factors.json`)

Two Sigma Factor Lens style: each factor is a fixed portfolio of ETFs from
`configs/fetch_data.json`. Factors are residualized level by level (level `n`
against every active factor at an earlier level), so a factor only carries the
risk the levels above it do not explain.

## Tickers used below

| Ticker | Tracks |
|---|---|
| SPY | US large-cap equity (S&P 500) |
| EFA | Developed-market ex-US, ex-Canada equity |
| EEM | Emerging-market equity |
| IWM | US small-cap equity (Russell 2000) |
| QQQ | US large-cap tech (Nasdaq 100) |
| IEF | 7-10y US Treasuries |
| TLT | 20y+ US Treasuries |
| SHY | 1-3y US Treasuries |
| HYG | US high-yield corporate bonds |
| LQD | US investment-grade corporate bonds |
| DBC | Broad commodities basket |
| TIP | US Treasury inflation-protected bonds |
| UUP | US dollar index (long-dollar) |
| FXY | Japanese yen |
| GLD | Gold |
| OIH | Oil-services equities |
| USO | Crude oil |
| XLI/XLB/XLY/XLF | Industrials / materials / discretionary / financials (cyclical sectors) |
| XLP/XLU/XLV | Staples / utilities / health care (defensive sectors) |

## Factors

| Level | Factor | Weights | Why |
|---|---|---|---|
| 1 core | Equity | 0.6 SPY + 0.3 EFA + 0.1 EEM | global equity market beta, cap-weighted US/developed/EM split |
| 1 core | Rates | IEF | duration / rates beta |
| 2 core | Credit | 0.5 HYG + 0.5 LQD | credit spread risk, residualized against Equity+Rates so it isolates spread, not duration or equity beta |
| 2 core | Commodities | DBC | broad commodity beta |
| 3 secondary | Emerging Markets | EEM | EM-specific risk left over after the Equity factor (which already contains 10% EEM) |
| 3 secondary | Foreign Currency | -UUP | non-USD currency risk |
| 3 secondary | Local Inflation | TIP | breakeven-inflation-like risk, net of Rates |
| 3 secondary | US vs Intl | SPY - EFA | US vs. developed-ex-US relative equity, net of the global Equity factor |
| 4 granular | Size | IWM - SPY | small- vs large-cap, off by default (`granular: false`) |
| 4 granular | Tech | QQQ - SPY | tech vs. broad market |
| 4 granular | Cyclical vs Defensive | mean(XLI,XLB,XLY,XLF) - mean(XLP,XLU,XLV) | cyclical vs. defensive sectors |
| 4 granular | Curve | TLT - SHY | yield-curve steepener/flattener, net of Rates |
| 4 granular | Energy | mean(XLE,OIH,USO) | energy-specific commodity exposure |
| 4 granular | Precious Metals | GLD | gold, net of broad Commodities |
| 4 granular | Yen | FXY | yen-specific currency risk, net of the broad dollar factor |

Levels 1-3 are always on. Level 4 (`granular`) is a config toggle
(`factor_model.granular` in the backtest configs) for a finer breakdown; off
by default because at that granularity ETF-specific noise starts to compete
with the factor signal.
