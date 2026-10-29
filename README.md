# 畔 相場の鏡

いまの相場が過去150年のいつに似ているかを探し、似た局面のその後と、
その鏡が未来を当てられるかの答え合わせを並べるサイト。

## 見かた
このリポジトリを手元に clone して、`site/index.html` をブラウザで開くだけで見られます（サーバー不要）。

```
git clone https://github.com/odennopython/stock_site
```

## 構成
- `engine/data.py`      公開データ（Shiller S&P 500、CPI、10年金利、金）を月次パネルに
- `engine/features.py`  25の状態指標と、その後のリターン・下落
- `engine/encoder.py`   NumPy製のノイズ除去自己符号化器（8次元の「指紋」）
- `engine/analogs.py`   近傍探索、エピソード抽出
- `engine/backtest.py`  1921年以降のウォークフォワード検証 → `data/backtest.json`
- `engine/build_site.py` サイト用 `site/data.json`
- `site/index.html`     静的サイト（`data.js` を読むので、ファイルを直接開いても動く）

## 更新
```
./fetch_data.sh
cd engine && python3 backtest.py && python3 build_site.py
```
