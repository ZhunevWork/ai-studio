# HYDRA Trading Assistant

This repository hosts a minimal prototype for the HYDRA trading assistant. It
includes indicator implementations and simple signal logic for research and
development of advanced trading strategies.

The `hydra_prototype.py` module provides:

- A collection of classic technical indicators (RSI, MACD, ADX, etc.)
- Feature construction utilities combining these indicators
- Basic regime classification and breakout/mean-reversion signals
- Simplified triple-barrier stop-loss and take-profit levels

The code is structured to be easily extended with additional market microstructure
features, order-flow metrics, and execution logic.

## Usage

Supply a DataFrame with `open`, `high`, `low`, `close`, and `volume` columns for
the desired timeframe and run:

```python
from hydra_prototype import signals, barriers
sig, feats = signals(df)
sl, tp = barriers(df, feats)
```

This prototype is a starting point for building the full HYDRA trading assistant.
