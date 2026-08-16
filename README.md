# trading-strategies

Graph and factor-based trading strategies, built on
[backtester](https://github.com/KaitoFuss/backtester)'s event-driven engine.

## What is this

`backtester` owns the engine, execution model, risk exits, and reporting.
This repo only adds new `Strategy` / `Portfolio` implementations against its
structural `Protocol`s — starting with a network-momentum signal (cross-asset
correlation graph → momentum score), inspired by
[Quantitativo's writeup](https://www.quantitativo.com/p/network-momentum) of
Poh, Lim, Zohren & Roberts (2022) [*Building Cross-Sectional Systematic
Strategies by Learning to Rank*](https://arxiv.org/abs/2012.07149).

The paper trades 60 continuous futures contracts, several of them
non-US-listed with same-day settlement well before the US close — the
replication attributes part of the paper's headline Sharpe to a timezone
lookahead bias from that mismatch. This repo trades a 26-name, four-asset-class
ETF universe instead: all US-listed, same-close, no roll-methodology
ambiguity, and no continuous-contract stitching to get wrong.

## Universe

| Class | Tickers |
|---|---|
| Equities | SPY, QQQ, IWM, EFA, EEM, VGK, EWJ |
| Fixed income | IEF, TLT, SHY, TIP, LQD, HYG, BND |
| Commodities | GLD, SLV, DBC, USO, UNG, DBA |
| Currencies | UUP, FXE, FXB, FXY, FXA, FXC |

`configs/fetch_data.json` starts `2007-04-18` — confirmed the earliest date
all 26 tickers have data on yfinance (`UNG` is the binding constraint; every
other name starts earlier).

## Quickstart

```bash
uv sync
uv run scripts/fetch_data.py configs/fetch_data.json
```

Pulls the universe above into one tidy Parquet file, `data/raw.parquet`,
using `backtester`'s `fetch_to_parquet` directly.

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

Optional: install pre-commit hooks so formatting/linting run on `git commit`:

```bash
uv run --with pre-commit pre-commit install
```

## License

MIT — see [LICENSE](LICENSE).
