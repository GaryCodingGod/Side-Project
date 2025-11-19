import yfinance as yf
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import requests 

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)

# ====== Discord Webhook ======
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1440667052640829522/i9XOM0CJiQT2VYXvk84i0Cx-wD4MKKfdD-i6tUgNsYCSPE4DA4p5okzYlxU4XCcNlI-E"


def send_to_discord(symbol, currency,
                    top5_outlier: pd.DataFrame,
                    top5_profit,
                    rolling_msg: str):
    # Outlier 文字
    outlier_lines = []
    for i, row in top5_outlier.reset_index(drop=True).iterrows():
        outlier_lines.append(f"{i+1}. {row['日期']} — L2 = {row['L2_norm']:.4f}")
    outlier_text = "\n".join(outlier_lines)

    # Profit 文字（top5_profit 是 list of (buy_date, sell_date, profit)）
    profit_lines = []
    for i, (b, s, p) in enumerate(top5_profit):
        profit_lines.append(f"{i+1}. 買：{b} → 賣：{s}，獲利：{p:.1f}")
    profit_text = "\n".join(profit_lines)

    message = (
        f"📊 **{symbol.upper()} 最近 6 個月股票分析結果（{currency}）**\n\n"
        f"📈 **Rolling Mean **\n{rolling_msg}\n\n"
        f"🔥 **最反常的 5 天（Outlier）**\n{outlier_text}\n\n"
        f"💰 **最獲利的 5 組買賣組合**\n{profit_text}"
    )

    try:
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        print(f"[Discord] 已送出 {symbol} 分析結果，狀態碼：{res.status_code}")
    except Exception as e:
        print("[Discord] 傳送失敗：", e)


# ================== 判斷台股 / 美股 ==================

def is_tw_symbol(symbol: str) -> bool:
    """純數字 or 結尾是 .TW 視為台股，其它視為美股"""
    s = symbol.strip().upper()
    return s.isdigit() or s.endswith(".TW")


def normalize_to_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    if s.isdigit():
        return s + ".TW"
    return s


# ================== 下載單一股票（yfinance） ==================

def fetch_single_stock_yf(symbol: str, period: str = "6mo") -> pd.DataFrame:
    yf_code = normalize_to_yf(symbol)
    print(f"\n抓取 {symbol} ({yf_code}) 最近 {period} 資料中...")

    df = yf.download(yf_code, period=period, interval="1d",
                     auto_adjust=False, progress=False)

    if df.empty:
        print(f"[警告] {symbol} 沒抓到任何資料")
        return pd.DataFrame()

    df = df.reset_index()

    if "Date" in df.columns:
        date_series = df["Date"]
    else:
        date_series = df.iloc[:, 0]

    out = pd.DataFrame()
    out["日期"] = pd.to_datetime(date_series).dt.strftime("%Y-%m-%d")
    out["成交股數"] = df["Volume"]
    out["開盤價"] = df["Open"]
    out["最高價"] = df["High"]
    out["最低價"] = df["Low"]
    out["收盤價"] = df["Close"]

    # 轉成數值
    num_cols = ["成交股數", "開盤價", "最高價", "最低價", "收盤價"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=num_cols).reset_index(drop=True)

    # 成交金額：「成交股數 * 收盤價」當近似值
    out["成交金額"] = out["成交股數"] * out["收盤價"]

    # 漲跌價差：今天收盤 - 昨天收盤（第一天沒有前一天，用 0 代替）
    out["漲跌價差"] = out["收盤價"].diff().fillna(0)

    # 成交筆數：yfinance 沒提供真實值，用 NaN 當佔位
    out["成交筆數"] = np.nan

    # 成交量 MA
    out["MA5"] = out["成交股數"].rolling(5).mean()
    out["MA20"] = out["成交股數"].rolling(20).mean()
    out["MA60"] = out["成交股數"].rolling(60).mean()

    # 收盤價 MA
    out["收盤價_MA5"] = out["收盤價"].rolling(5).mean()
    out["收盤價_MA20"] = out["收盤價"].rolling(20).mean()
    out["收盤價_MA60"] = out["收盤價"].rolling(60).mean()

    cols_order = [
        "日期",
        "成交股數",
        "成交金額",
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "漲跌價差",
        "成交筆數",
        "MA5",
        "MA20",
        "MA60",
        "收盤價_MA5",
        "收盤價_MA20",
        "收盤價_MA60",
    ]
    out = out[cols_order]

    csv_name = f"{symbol.upper()}_yf_last6m.csv"
    out.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"[CSV] 已存成 {csv_name}")

    return out


# ================== 建多股票收盤價表格 ==================

def build_close_price_table(symbols, period="6mo"):
    close_df = pd.DataFrame()
    data_map = {}

    for raw in symbols:
        sym = raw.strip()
        if not sym:
            continue

        df = fetch_single_stock_yf(sym, period=period)
        if df is not None and not df.empty:
            data_map[sym] = df

            currency = "TWD" if is_tw_symbol(sym) else "USD"
            label = f"{sym.upper()} ({currency})"

            s = df.set_index("日期")["收盤價"].rename(label)
            if close_df.empty:
                close_df = s.to_frame()
            else:
                close_df = close_df.join(s, how="outer")

    close_df = close_df.sort_index()
    return close_df, data_map


# ================== 畫多股票收盤價折線圖 ==================

def plot_multi_close(close_df: pd.DataFrame):
    if close_df.empty:
        print("沒有任何收盤價資料可畫圖。")
        return

    ax = close_df.plot(figsize=(10, 6))
    ax.set_title("多檔股票收盤價折線圖")
    ax.set_xlabel("日期")
    ax.set_ylabel("價格")
    plt.xticks(rotation=45)
    plt.tight_layout()

    img_name = "multi_stocks.png"
    plt.savefig(img_name)
    plt.close()
    print(f"[圖檔] 已輸出折線圖：{img_name}")


# ================== Outlier & Most profit ==================

def normalization(df: pd.DataFrame):
    """
    使用三個特徵：
    1. 成交股數
    2. 收盤價 - 開盤價
    3. 最高價 - 最低價
    做 z-score normalization
    """
    df["成交股數_差"] = df["成交股數"] - df["成交股數"].mean()
    std_vol = df["成交股數"].std()
    if std_vol == 0:
        std_vol = 1.0
    df["成交股數_norm"] = df["成交股數_差"] / std_vol

    df["收盤價差"] = df["收盤價"] - df["開盤價"]
    df["收盤價差_差"] = df["收盤價差"] - df["收盤價差"].mean()
    std_cd = df["收盤價差"].std()
    if std_cd == 0:
        std_cd = 1.0
    df["收盤價差_norm"] = df["收盤價差_差"] / std_cd

    df["最高低價差"] = df["最高價"] - df["最低價"]
    df["最高低價差_差"] = df["最高低價差"] - df["最高低價差"].mean()
    std_hl = df["最高低價差"].std()
    if std_hl == 0:
        std_hl = 1.0
    df["最高低價差_norm"] = df["最高低價差_差"] / std_hl


def returnTop5Outlier(df: pd.DataFrame):
    normalization(df)

    df["L2_norm"] = np.sqrt(
        df["成交股數_norm"] ** 2 +
        df["收盤價差_norm"] ** 2 +
        df["最高低價差_norm"] ** 2
    )

    top5 = df.nlargest(5, "L2_norm")[["日期", "L2_norm"]]

    print("\n---- 最反常的 5 天（Outlier）----")
    for i, row in top5.reset_index(drop=True).iterrows():
        print(f"{i+1}. {row['日期']}  L2 = {row['L2_norm']:.6f}")

    return top5


def findMaxProfit(data, MAXPROFIT):
    length = len(data)
    maxProfit = -sys.float_info.max
    buyDay = -1
    sellDay = -1

    for i in range(length - 1):
        for j in range(i + 1, length):
            profit = data[j] - data[i]
            if profit > maxProfit and profit < MAXPROFIT:
                buyDay = i
                sellDay = j
                maxProfit = profit

    return buyDay, sellDay, maxProfit


def findTop5ProfitDay(df: pd.DataFrame):
    prices = df["收盤價"].values.astype(float)
    dates = df["日期"].values

    MAX = sys.float_info.max
    top5Buy = [None] * 5
    top5Sell = [None] * 5
    top5Profit = [None] * 5

    for k in range(5):
        buyIdx, sellIdx, profit = findMaxProfit(prices, MAX)
        if buyIdx == -1:
            break
        top5Buy[k] = buyIdx
        top5Sell[k] = sellIdx
        top5Profit[k] = profit
        MAX = profit

    print("\n---- 最獲利的 5 組買賣組合 ----")
    for rank in range(5):
        if top5Buy[rank] is None:
            continue
        b = top5Buy[rank]
        s = top5Sell[rank]
        p = top5Profit[rank]
        print(f"Top {rank+1} : 買 {dates[b]} → 賣 {dates[s]}，獲利 {p:.1f}")

    result = []
    for rank in range(5):
        if top5Buy[rank] is None:
            continue
        b = top5Buy[rank]
        s = top5Sell[rank]
        p = top5Profit[rank]
        result.append((dates[b], dates[s], p))

    return result



def last_valid(df, col):
    s = df[col].dropna()
    if s.empty:
        return None
    return s.iloc[-1]


def build_rolling_summary(df: pd.DataFrame) -> str:
    v5 = last_valid(df, "MA5")
    v20 = last_valid(df, "MA20")
    v60 = last_valid(df, "MA60")

    c5 = last_valid(df, "收盤價_MA5")
    c20 = last_valid(df, "收盤價_MA20")
    c60 = last_valid(df, "收盤價_MA60")

    def fmt(x, digits=2):
        if x is None or pd.isna(x):
            return "N/A"
        return f"{x:.{digits}f}"

    lines = []
    if v5 is not None:
        lines.append(
            f"成交量 MA5 / MA20 / MA60：{fmt(v5,0)} / {fmt(v20,0)} / {fmt(v60,0)}"
        )
    if c5 is not None:
        lines.append(
            f"收盤價 MA5 / MA20 / MA60：{fmt(c5)} / {fmt(c20)} / {fmt(c60)}"
        )

    if not lines:
        return "（資料天數不足，尚無 MA 資訊）"

    return "\n".join(lines)



def main():
    print("範例：2330 2317 AAPL MSFT ...")
    raw = input("請輸入台/美股票代碼（用空白分開）：").strip()

    raw = raw.replace("，", " ").replace(",", " ")
    symbols = [s for s in raw.split() if s]

    if not symbols:
        print("沒有輸入任何股票代碼，結束。")
        return

    close_df, data_map = build_close_price_table(symbols, period="6mo")

    for sym in symbols:
        df0 = data_map.get(sym)
        if df0 is None or df0.empty:
            continue

        print("\n=== 原始價格表（前 10 列）:", sym, "===")
        print(df0.head(10))

        print(f"\n{sym}：成交股數 5 / 20 / 60 日 MA（前 10 筆）")
        print(df0[["日期", "成交股數", "MA5", "MA20", "MA60"]].head(10))

        print(f"\n{sym}：收盤價 5 / 20 / 60 日 MA（前 10 筆）")
        print(df0[["日期", "收盤價", "收盤價_MA5", "收盤價_MA20", "收盤價_MA60"]].head(10))

        print("\n===== 這段時間的價格分布 =====")
        price_cols = ["開盤價", "最高價", "最低價", "收盤價"]
        print(df0[price_cols].describe())

    plot_multi_close(close_df)

    for sym in symbols:
        df = data_map.get(sym)
        if df is None or df.empty:
            continue

        print("\n" + "=" * 60)
        currency = "新台幣 TWD" if is_tw_symbol(sym) else "美元 USD"
        print(f"股票：{sym.upper()}  |  貨幣：{currency}")

        df_analyse = df.copy()
        top5_outlier = returnTop5Outlier(df_analyse)
        top5_profit = findTop5ProfitDay(df_analyse)

        rolling_msg = build_rolling_summary(df)

        if DISCORD_WEBHOOK_URL and "discord.com" in DISCORD_WEBHOOK_URL:
            send_to_discord(sym, currency, top5_outlier, top5_profit, rolling_msg)


if __name__ == "__main__":
    main()
