import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_transaction_table(bought, sold, bought_date, sold_date):
    transaction_data = {'Type': [], 'Date': [], 'Price': []}

    for idx in range(max(len(bought), len(sold))):
        if idx < len(bought):
            transaction_data['Type'].append('Bought')
            transaction_data['Date'].append(bought_date[idx])
            transaction_data['Price'].append(bought[idx])
        else:
            transaction_data['Type'].append('')
            transaction_data['Date'].append('')
            transaction_data['Price'].append('')

        if idx < len(sold):
            transaction_data['Type'].append('Sold')
            transaction_data['Date'].append(sold_date[idx])
            transaction_data['Price'].append(sold[idx])
        else:
            transaction_data['Type'].append('')
            transaction_data['Date'].append('')
            transaction_data['Price'].append('')

    transaction_df = pd.DataFrame(transaction_data)
    return transaction_df


def calc_ema(data, N):
    up = data[0]
    down = 1
    alpha = 2 / (N + 1)
    for i in range(1, N):
        up += data[i - 1] * (1 - alpha) ** (i - 1)
        down += (1 - alpha) ** (i - 1)

    return up / down


def calc_macd(close_price):
    ema_values26 = np.zeros_like(close_price)
    ema_values12 = np.zeros_like(close_price)
    macd = np.zeros_like(close_price)
    macd_signal_line = np.zeros_like(close_price)

    for i in range(11, len(close_price)):
        if i >= 11:
            if i == 11:
                ema_values12[i] = calc_ema(close_price[0:11], 12)
            else:
                ema_values12[i] = calc_ema(close_price[(i - 11):i], 12)
        if i >= 25:
            if i == 25:
                ema_values26[i] = calc_ema(close_price[0:25], 26)
            else:
                ema_values26[i] = calc_ema(close_price[(i - 25):i], 26)

            macd[i] = ema_values12[i] - ema_values26[i]

        if i >= 34:
            macd_signal_line[i] = calc_ema(macd[(i - 8):i], 9)

    return macd, macd_signal_line


def plot_stock(name, dates, close_price, start_position):
    plt.figure(figsize=(1920 / 100, 700 / 100), dpi=100)
    start = start_position
    plt.plot(dates[start:], close_price[start:])
    plt.title(name)
    plt.xlabel("Dates")
    plt.ylabel("Prices [zł]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_macd(macd, macd_signal_line, macd_buy_sell, dates, start_position, cross_points=None):
    if cross_points is None:
        cross_points = []
    start = start_position

    # MACD Line and Signal Line
    plt.figure(figsize=(1920 / 100, 700 / 100), dpi=100)

    plt.plot(dates[start:], macd[start:], label='MACD', color='blue')
    plt.plot(dates[start:], macd_signal_line[start:], label='MACD Signal Line', color='red')

    macd_bar = np.zeros_like(macd_signal_line)
    for i in range(start, len(macd_signal_line)):
        macd_bar[i] = macd[i] - macd_signal_line[i]

    plt.bar(dates[start:], macd_bar[start:])
    if len(cross_points) == 0:
        cross_points_buy = [i for i in range(len(macd[start:])) if macd_buy_sell[start:][i][1] == "buy"]
        cross_points_sell = [i for i in range(len(macd[start:])) if macd_buy_sell[start:][i][1] == "sell"]

    else:
        cross_points_buy = cross_points[0]
        cross_points_sell = cross_points[1]
    if len(cross_points) == 0:
        plt.scatter(np.array(dates[start:])[cross_points_buy], np.array(macd[start:])[cross_points_buy],
                    color='yellow',
                    marker='^', s=100, edgecolors='black', label='Cross Points BUY')

        plt.scatter(np.array(dates[start:])[cross_points_sell], np.array(macd[start:])[cross_points_sell],
                    color='purple',
                    marker='v', s=100, edgecolors='black', label='Cross Points SELL')
    else:
        plt.vlines(np.array(dates[start:])[cross_points_buy],
                   ymin=np.min(macd),
                   ymax=np.max(macd),
                   color='yellow',
                   linestyles='dashed',
                   linewidth=2,
                   label='Cross Points BUY')

        plt.vlines(np.array(dates[start:])[cross_points_sell],
                   ymin=np.min(macd),
                   ymax=np.max(macd),
                   color='purple',
                   linestyles='dashed',
                   linewidth=2,
                   label='Cross Points SELL')

    plt.title('MACD')
    plt.xlabel('Dates')
    plt.ylabel('Indicator value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def macd_buy_sell(macd, macd_signal_line):
    histogram = [macd[i] - macd_signal_line[i] for i in range(len(macd))]
    buy_sell = [(0, "-")]
    for i in range(1, len(macd_signal_line)):
        if macd[i] > macd_signal_line[i] and macd[i - 1] < macd_signal_line[i - 1] and histogram[i] > 0:
            buy_sell.append((i, "buy"))
        elif macd[i] < macd_signal_line[i] and macd[i - 1] > macd_signal_line[i - 1] and histogram[i] < 0:
            buy_sell.append((i, "sell"))
        else:
            buy_sell.append((i, "-"))

    return buy_sell


def stooq_bot(name, macd, macd_signal_line, macd_buy_sell, close_price, dates, starting_day, starting_capital):
    dates_r = dates[starting_day:]
    close_price_r = close_price[starting_day:]
    macd_buy_sell_r = macd_buy_sell[starting_day:]
    macd_r = macd[starting_day:]
    bought = []
    sold = []
    bought_date = []
    sold_date = []
    bought_price = []
    sold_price = []

    profit = starting_capital

    buy = True
    for i in range(len(dates_r)):
        if buy:
            if (macd_buy_sell_r[i][1] == "buy" and
                    macd_r[i] < 0 and
                    all(macd_r[j] < macd_r[j + 1] for j in range(i - 3, i))):
                profit -= 1000 * close_price_r[i]
                bought.append(i)
                bought_date.append(np.datetime_as_string(dates_r[i], unit='D'))
                bought_price.append(close_price_r[i])
                print("Bought at ", np.datetime_as_string(dates_r[i], unit='D'), " on ", close_price_r[i])
                buy = False
        else:
            if (macd_buy_sell_r[i][1] == "sell" and
                    macd_r[i] > 0 and
                    all(macd_r[j] > macd_r[j + 1] for j in range(i - 2, i))):
                profit += 1000 * close_price_r[i]
                sold.append(i)
                sold_date.append(np.datetime_as_string(dates_r[i], unit='D'))
                sold_price.append(close_price_r[i])

                print("Sold at ", np.datetime_as_string(dates_r[i], unit='D'), " on ", close_price_r[i])
                buy = True

    plot_stock(name, dates, close_price, starting_day)
    plot_macd(macd, macd_signal_line, macd_buy_sell_r, dates, starting_day, [[bought], [sold]])

    print(create_transaction_table(bought_price, sold_price, bought_date, sold_date))
    return profit


def init_macd(name, csv_file):
    df = pd.read_csv(csv_file)

    dates = pd.to_datetime(df['Data']).to_numpy()
    close_price = df['Zamkniecie'].to_numpy()
    macd, macd_signal_line = calc_macd(close_price)
    buy_sell = macd_buy_sell(macd, macd_signal_line)
    plot_stock(name, dates, close_price, 0)
    plot_macd(macd, macd_signal_line, buy_sell, dates, len(macd_signal_line) - 100)

    starting_day = len(close_price) - 1200
    starting_capital = 100000
    profit = stooq_bot(name, macd, macd_signal_line, buy_sell, close_price, dates, starting_day, starting_capital)
    print("Start date: ", np.datetime_as_string(dates[starting_day], unit='D'))
    print("End date: ", np.datetime_as_string(dates[-1], unit='D'))
    print("Profit: ", profit - starting_capital)
