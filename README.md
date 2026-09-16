# trading-strategies

Graph and factor trading strategies. Built on
[backtester](https://github.com/KaitoFuss/backtester)'s event-driven engine.

## What is this

`backtester` owns engine, execution model, risk exits, reporting. This repo
adds new `Strategy` / `Portfolio` implementations against its structural
`Protocol`s only — nothing else. Starts with network-momentum signal
(cross-asset correlation graph → momentum score), inspired by
[Quantitativo's writeup](https://www.quantitativo.com/p/network-momentum) of
Poh, Lim, Zohren & Roberts (2022) [*Building Cross-Sectional Systematic
Strategies by Learning to Rank*](https://arxiv.org/abs/2012.07149).

Paper trades 60 continuous futures. Several non-US-listed, settle same-day
before US close. Replication blames part of paper's headline Sharpe on
timezone lookahead bias from that mismatch. This repo trades 61-name,
four-asset-class ETF universe instead: all US-listed, same-close. No
roll-methodology ambiguity. No continuous-contract stitching to get wrong.

## Universe

| Class | Tickers |
|---|---|
| Equities — broad | SPY, QQQ, IWM, EFA, EEM, VGK, EWJ |
| Equities — US sectors | XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU |
| Equities — US industries | SMH, IBB, KRE, XHB, GDX, OIH, IYT, IYR |
| Equities — developed countries | EWC, EWA, EWU, EWL, EWP, EWI, EWS |
| Equities — emerging countries | EWZ, FXI, EWY, EWT, EWH, EWW, EZA, EWM |
| Fixed income | IEF, TLT, SHY, TIP, LQD, HYG, BND, MBB, IGIB, PFF |
| Commodities | GLD, SLV, DBC, USO, UNG, DBA |
| Currencies | UUP, FXE, FXB, FXY, FXA, FXC, FXF |

`configs/fetch_data.json` starts `2007-04-18` — earliest date all 61 tickers
have data on yfinance, confirmed directly. `UNG` binding constraint; every
other name starts earlier.

Selection rules for names beyond the original 26: yfinance history starts on
or before `2007-04-18`, median 2025 daily dollar volume of roughly $5M or
more, and absolute daily-return correlation below ~0.95 with every other name
(drops e.g. XLK≈QQQ, EWG≈VGK, VNQ≈IYR, TLH≈TLT, IEI≈IEF). One pre-existing
pair exceeds that bar: EFA/VGK at 0.98.

Tradeoff: 30 of the 35 additions are equities, so the equity factor
dominates. Effective breadth (participation ratio of the return-correlation
eigenvalues, 2007-04-18 → 2025) falls from 5.4 to 4.2. Diversity comes from
the risk model, which must control the equity exposure. Screen covers only
tickers still listed on yfinance, so it carries survivorship bias.

## Quickstart

```bash
uv sync
uv run scripts/fetch_data.py configs/fetch_data.json
```

Pulls universe above into one tidy Parquet file, `data/raw.parquet`. Uses
`backtester`'s `fetch_to_parquet` directly, no reimplementation.

## Project structure

```
src/trading_strategies/  # library source (src layout)
tests/                    # tests, mirroring src/trading_strategies/ module paths
scripts/                  # CLI entry points
configs/                  # fetch/backtest configs
pyproject.toml            # project metadata, dependencies, and tool config
```

## Development

```bash
uv sync                      # install/sync deps
uv run ruff format .         # format
uv run ruff check . --fix    # lint
uv run mypy .                # strict type check
uv run pytest                # tests (coverage gate: --cov-fail-under=95)
```

Optional: install pre-commit hooks, formatting/linting run on `git commit`:

```bash
uv run --with pre-commit pre-commit install
```

## License

MIT — see [LICENSE](LICENSE).
