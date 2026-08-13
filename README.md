# KOSPI200 SSL Hybrid Backtest

현재 KOSPI200 구성종목을 백테스트 전 기간에 고정해서 사용하는 SSL Hybrid 롱 전략 백테스트입니다.
요청한 방식대로 **생존편향을 보정하지 않습니다.**

## 기본 전략

- Universe: 현재 KOSPI200 구성종목
- Baseline: HMA(60)
- SSL2: JMA(5), phase=3, power=1
- Exit: HMA(15)
- ATR: WMA(True Range, 14)
- ATR continuation criteria: 0.9
- Entry: `SSL2 BUY Continuation`이 새로 발생한 날 종가 신호 → 다음 거래일 시가 매수
- Exit: 종가가 `sslExit`을 하향 돌파한 날 종가 신호 → 다음 거래일 시가 매도
- Short: 사용하지 않음
- 기본 거래비용: 0

## GitHub Actions에서 실행

1. 이 저장소를 GitHub에 업로드합니다.
2. GitHub 저장소의 **Actions** 탭으로 이동합니다.
3. **KOSPI200 SSL Hybrid Backtest**를 선택합니다.
4. **Run workflow**를 누릅니다.
5. 시작일/종료일/수수료 등을 입력하고 실행합니다.
6. 완료 후 해당 Action 실행 화면 아래의 **Artifacts**에서 `ssl-hybrid-backtest-results`를 받습니다.

기본 실행 기간은 `2023-08-10 ~ 2026-08-10`입니다.

## 로컬 실행

```bash
pip install -r requirements.txt
python kospi200_ssl_hybrid_backtest.py --online
```

예시:

```bash
python kospi200_ssl_hybrid_backtest.py \
  --online \
  --start 2023-08-10 \
  --end 2026-08-10 \
  --fee-side 0.001 \
  --baseline-len 60 \
  --ssl2-len 5 \
  --exit-len 15 \
  --atr-len 14 \
  --atr-crit 0.9 \
  --out output
```

## 출력 파일

`output/` 폴더에 아래 파일이 생성됩니다.

- `summary.csv`: 전체 동일가중 포트폴리오 및 횡단면 요약
- `by_stock.csv`: 종목별 수익률, CAGR, MDD, Sharpe, 승률 등
- `trades.csv`: 모든 매매 기록
- `portfolio_daily.csv`: KOSPI200 동일가중 전략 포트폴리오 일별 성과

## 주의

현재 시점의 KOSPI200 구성종목을 과거 전체 구간에 그대로 적용하므로 생존편향이 존재합니다. 이는 의도된 설정입니다.
데이터 소스 차이와 지표 초기화 방식 때문에 TradingView Pine Script와 아주 작은 수치 차이가 발생할 수 있습니다.
