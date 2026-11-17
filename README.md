# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Daniikhwan/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Daniikhwan/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch concurrent viewers hour by hour (hours 2–24) from the hour-1 print. The forecaster learns the launch-day shape from the case's own comparables — 42 past big launches: for each hour it regresses log viewers on log hour-1 viewers across the comparables (big launches grow far less through the day than small ones do), shrinks the per-hour slopes toward their mean, and for very large hour-1 prints blends toward the median profile of the largest launches, which stay roughly flat. Quantile spreads come from the regression residuals, measured robustly. Practice pinball loss 0.169 against the house baseline's 0.387, with 90% coverage 0.96 and a typical miss of 30%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The program reads every `<id>.in` case in the input directory and writes one `<id>.out` forecast per case to the output directory. Python 3.13 standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vnsny7za&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
