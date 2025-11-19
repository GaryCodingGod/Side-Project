import requests
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from dateutil.relativedelta import relativedelta

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)

#連接到 Discord 的 Webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1440251632293253151/LnVfDqy26lAVITE3Gg5NvV1bC5AvRZt4c_8B208j4H7FWrIpnVJFu1u7ECURhme7u-m_"

def send_to_discord(stock_no, top5_outlier, top5_profit):
    # 整理 Outlier 資料
    outlier_text = "\n".join(
        [f"{i+1}. {row['日期']} — L2 = {row['L2_norm']:.4f}"
         for i, row in top5_outlier.reset_index(drop=True).iterrows()]
    )

    # 整理 Top5 profit 資料
    profit_lines = []
    for i, (buy, sell, profit) in enumerate(top5_profit):
        profit_lines.append(f"{i+1}. 買：{buy} → 賣：{sell}，利潤：{profit:.2f}")
    profit_text = "\n".join(profit_lines)

    # 最終送到 Discord 的文字
    message = (
        f"📊 **{stock_no} 最近 6 個月股票分析結果**\n\n"
        f"🔥 最反常的 5 天（Outliers）**\n{outlier_text}\n\n"
        f"💰 最獲利的 5 組買賣組合**\n{profit_text}"
    )

    # 送出
    try:
        res = requests.post(DISCORD_WEBHOOK_URL, json={"content": message})
        print("已傳送到 Discord，狀態碼：", res.status_code)
    except Exception as e:
        print("傳送到 Discord 失敗：", e)


# ================== Fetch Data ==================
def get_month_starts_last_n_months(n=2):
    today = datetime.today()
    first_this_month = today.replace(day=1)

    dates = []
    d = first_this_month
    for _ in range(n):
        dates.append(d.strftime("%Y%m01"))
        d -= relativedelta(months=1)

    dates.sort()
    return dates


def fetch_month_df(date_str, stock_no="2330"):
    url = (
        "https://www.twse.com.tw/exchangeReport/STOCK_DAY"
        f"?response=csv&date={date_str}&stockNo={stock_no}"
    )
    r = requests.get(url)
    r.encoding = "utf-8"

    lines = [
        line.strip()
        for line in r.text.splitlines()
        if line.strip().startswith('"')
    ]

    if not lines:
        print(f"[警告] {date_str} {stock_no} 沒抓到任何資料")
        return None

    rows = []
    for line in lines:
        if line.endswith('",'):
            line = line[:-2]
        if line.startswith('"'):
            line = line[1:]
        parts = line.split('","')
        rows.append(parts)

    raw_df = pd.DataFrame(rows)

    header = raw_df.iloc[0]
    df = raw_df.iloc[1:].reset_index(drop=True)
    df.columns = header

    if df.shape[1] < 9:
        print(f"[警告] {date_str} {stock_no} 欄位數只有 {df.shape[1]}，略過")
        return None

    df = df.iloc[:, :9]
    df.columns = [
        "日期",
        "成交股數",
        "成交金額",
        "開盤價",
        "最高價",
        "最低價",
        "收盤價",
        "漲跌價差",
        "成交筆數",
    ]

    return df


def get_last_n_months_data(stock_no="2330", n=6): #合併最近 n 個月全部資料(可改參數)
    dates = get_month_starts_last_n_months(n)
    dfs = []

    for d in dates:
        print(f"抓取 {stock_no} {d} 當月資料中...")
        mdf = fetch_month_df(d, stock_no=stock_no)
        if mdf is not None and not mdf.empty:
            dfs.append(mdf)

    if not dfs:
        print(f"最近幾個月完全沒抓到 {stock_no} 的任何資料")
        return pd.DataFrame()

    all_df = pd.concat(dfs, ignore_index=True)
    return all_df


def clean_df(df: pd.DataFrame):
    num_cols = ["成交股數", "成交金額", "成交筆數"]
    for col in num_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    price_cols = ["開盤價", "最高價", "最低價", "收盤價"]
    for col in price_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=num_cols + price_cols)
    df = df.reset_index(drop=True)
    return df


# ================== Outlier ==================

def normalization(df: pd.DataFrame):
    """
    使用作業指定的三個特徵：
    1. daily volume (成交股數)
    2. closing price - opening price (收盤價 - 開盤價)
    3. highest price - lowest price (最高價 - 最低價)

    做 z-score normalization： (Xi - Mean(X)) / Std(X)
    """
    # 特徵 1：成交股數
    df["成交股數_差"] = df["成交股數"] - df["成交股數"].mean()
    std_vol = df["成交股數"].std()
    if std_vol == 0:
        std_vol = 1.0
    df["成交股數_norm"] = df["成交股數_差"] / std_vol

    # 特徵 2：收盤價 - 開盤價
    df["收盤價差"] = df["收盤價"] - df["開盤價"]
    df["收盤價差_差"] = df["收盤價差"] - df["收盤價差"].mean()
    std_cd = df["收盤價差"].std()
    if std_cd == 0:
        std_cd = 1.0
    df["收盤價差_norm"] = df["收盤價差_差"] / std_cd

    # 特徵 3：最高價 - 最低價
    df["最高低價差"] = df["最高價"] - df["最低價"]
    df["最高低價差_差"] = df["最高低價差"] - df["最高低價差"].mean()
    std_hl = df["最高低價差"].std()
    if std_hl == 0:
        std_hl = 1.0
    df["最高低價差_norm"] = df["最高低價差_差"] / std_hl


def returnTop5Outlier(df: pd.DataFrame):
    """
    先做 normalization，接著以三個標準化後的特徵計算 L2-norm，
    找出差異最大的 5 天（由大到小）。
    """
    normalization(df)

    # L2-norm = sqrt( x1^2 + x2^2 + x3^2 )
    df["L2_norm"] = np.sqrt(
        df["成交股數_norm"] ** 2 +
        df["收盤價差_norm"] ** 2 +
        df["最高低價差_norm"] ** 2
    )

    top5 = df.nlargest(5, "L2_norm")[["日期", "L2_norm"]]

    print("\n================ implement 4:Outlier  Special days ================")
    print("Special days: Compared to average, what days are most unusual?\n")
    print("TOP 5 差異最大的日子（由遠到近排序）：")
    for i, row in top5.reset_index(drop=True).iterrows():
        print(f"{row['日期']} {row['L2_norm']:.6f}")

    return top5


# ================== Most profit ==================

def findMaxProfit(data, MAXPROFIT):
    """
    在 data（價格 list/ndarray）中找出：
    在 profit < MAXPROFIT 的條件下，可以得到的最大 profit 與其買賣日 index

    回傳：buy_index, sell_index, maxProfit
    若找不到則回傳 -1, -1, 最小值
    """
    length = len(data)
    maxProfit = -sys.float_info.max
    buyDay = -1
    sellDay = -1

    for i in range(length - 1):        # 買入日
        for j in range(i + 1, length): # 賣出日
            profit = data[j] - data[i]
            if profit > maxProfit and profit < MAXPROFIT:
                buyDay = i
                sellDay = j
                maxProfit = profit

    return buyDay, sellDay, maxProfit


def findTop5ProfitDay(df: pd.DataFrame):
    """
    使用收盤價作為當天價格，列出 Top 5 獲利最高的：
    買入日期、賣出日期、獲利
    （時間複雜度 O(n^2)）
    """
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
        MAX = profit   # 下一輪 profit 需 < 前一輪找到的最大 profit

    print("\n================ implement 5:Most profit =================")
    print("TOP 5 max profit: 買入日期、賣出日期、獲益（Top1 → Top5）\n")

    for rank in range(5):
        if top5Buy[rank] is None:
            continue
        b = top5Buy[rank]
        s = top5Sell[rank]
        p = top5Profit[rank]
        print(f"Top {rank+1} : {dates[b]}, {dates[s]}, {p:.1f}")

    result = []
    for rank in range(5):
        if top5Buy[rank] is None:
            continue
        b = top5Buy[rank]
        s = top5Sell[rank]
        p = top5Profit[rank]
        result.append((dates[b], dates[s], p))

    return result



def main():
    stock_no = input("請輸入股票代碼（預設 2330）：").strip()
    if stock_no == "":
        stock_no = "2330"

    # 抓最近 6 個月資料
    df = get_last_n_months_data(stock_no=stock_no, n=6)

    print("抓到原始資料筆數 =", len(df))
    if df.empty:
        return

    df = clean_df(df)
    print("清洗後資料筆數 =", len(df))

    # 存成 csv
    csv_name = f"{stock_no}_last6m.csv"
    df.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"已存成 {csv_name}")

    # 收盤價 5 日均線
    df["收盤價_MA5"] = df["收盤價"].rolling(5).mean()
    # 成交股數 5 日均線：日期｜成交股數｜MA5
    df["MA5"] = df["成交股數"].rolling(5).mean()

    print("\n===== implement 1：df.head() & MA5 =====")
    print(df.head())

    print(f"\n{stock_no}：成交股數 5 日 MA（前 10 筆）")
    print(df[["日期", "成交股數", "MA5"]].head(10))

    print("\n===== implement 2：df.describe() =====")
    print(df[["開盤價", "最高價", "最低價", "收盤價"]].describe())

    # 價格變化 = 當日收盤價 - 前一天收盤價
    df["dA"] = df["收盤價"] - df["收盤價"].shift(1)

    # 百分比變化 = 價格變化 / 前一天收盤價
    df["dB"] = df["dA"] / df["收盤價"].shift(1) * 100

    idx_abs = df["dA"].abs().idxmax()
    row_abs = df.loc[idx_abs]

    idx_pct = df["dB"].abs().idxmax()
    row_pct = df.loc[idx_pct]

    print("\n===== implement 3：最大單日變化 =====")
    print(" 💵以 absolute amount 找價格變化最大的一天💵 ")
    print("日期：", row_abs["日期"])
    print("變化的數值（dA）：", row_abs["dA"])
    print("當日最高價：", row_abs["最高價"])
    print("當日成交股數：", row_abs["成交股數"])

    print("\n 💵以 percentage 找價格變化最大的一天💵 ")
    print("日期：", row_pct["日期"])
    print("變化的百分比（dB，%）：", row_pct["dB"])
    print("當日最高價：", row_pct["最高價"])
    print("當日成交股數：", row_pct["成交股數"])

    print("\n(a) 價格變化最大那天的完整資料：")
    print(df.loc[[idx_abs]])

    print("\n(b) 百分比變化最大那天的完整資料：")
    print(df.loc[[idx_pct]])

    top5_outlier = returnTop5Outlier(df)

    top5_profit = findTop5ProfitDay(df)

    send_to_discord(stock_no, top5_outlier, top5_profit)


if __name__ == "__main__":
    main()
