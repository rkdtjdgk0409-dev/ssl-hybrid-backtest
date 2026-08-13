# KOSPI200 SSL Hybrid Backtest

TradingView SSL Hybrid 로직을 Python으로 옮긴 KOSPI200 백테스트입니다.

## Universe
- 현재 KOSPI200 구성종목 200개를 기준으로 전체 과거 기간을 테스트합니다.
- 과거 편입/편출을 추적하지 않으므로 생존편향을 보정하지 않습니다.
- GitHub Actions에서는 KRX 로그인을 요구하지 않는 Naver Finance 구성종목 페이지를 우선 사용합니다.
- Naver 실패 시 pykrx -> Hankyung 순으로 자동 fallback 합니다.

## Strategy defaults
- Baseline: HMA 60
- SSL2: JMA 5
- Exit: HMA 15
- ATR: WMA 14
- ATR continuation criterion: 0.9
- 매수 신호가 종가에 발생하면 다음 거래일 시가 매수
- Exit Long 신호가 종가에 발생하면 다음 거래일 시가 매도

## GitHub Actions
1. 이 폴더 내용을 GitHub 저장소 루트에 업로드합니다.
2. Actions 탭에서 `KOSPI200 SSL Hybrid Backtest`를 선택합니다.
3. `Run workflow`를 누릅니다.
4. 완료 후 `ssl-hybrid-backtest-results` artifact를 내려받습니다.

## Output
- `output/summary.csv`
- `output/by_stock.csv`
- `output/trades.csv`
- `output/portfolio_daily.csv`

## Local run
```bash
pip install -r requirements.txt
python kospi200_ssl_hybrid_backtest.py --online --out output
```

## KOSPI200 constituent-count note
The online universe loader accepts a current Naver KOSPI200 snapshot containing **195-200 unique stocks**. This avoids false failures when the live index/replication basket temporarily contains 199 listed constituents. A materially incomplete scrape (<195) still fails instead of silently running a truncated backtest.

The backtest intentionally uses the **current constituent snapshot for the whole historical test window** (no survivorship-bias adjustment), as requested.
