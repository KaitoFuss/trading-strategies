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
timezone lookahead bias from that mismatch. This repo trades 26-name,
four-asset-class ETF universe instead: all US-listed, same-close. No
roll-methodology ambiguity. No continuous-contract stitching to get wrong.

## Universe

| Class | Tickers |
|---|---|
| Equities | SPY, QQQ, IWM, EFA, EEM, VGK, EWJ |
| Fixed income | IEF, TLT, SHY, TIP, LQD, HYG, BND |
| Commodities | GLD, SLV, DBC, USO, UNG, DBA |
| Currencies | UUP, FXE, FXB, FXY, FXA, FXC |

`configs/fetch_data.json` starts `2007-04-18` — earliest date all 26 tickers
have data on yfinance, confirmed directly. `UNG` binding constraint; every
other name starts earlier.

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
