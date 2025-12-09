import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import ta # Technical Analysis library

# --- 1. Сбор данных ---
def download_stock_data(ticker, start_date, end_date):
    """
    Загружает исторические данные акций с Yahoo Finance.
    """
    print(f"Загрузка данных для {ticker} с {start_date} по {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"Не удалось загрузить данные для {ticker}. Проверьте тикер и даты.")
    print("Данные успешно загружены.")
    return data

# --- 2. Генерация признаков (Feature Engineering) ---
def create_features(df):
    """
    Создает технические индикаторы в качестве признаков.
    Использует библиотеку `ta` для удобства.
    """
    print("Генерация признаков...")
    # Признаки на основе скользящих средних
    df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA_30'] = ta.trend.sma_indicator(df['Close'], window=30)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

    # Признак на основе RSI (индекс относительной силы)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # Признаки на основе MACD (схождение/расхождение скользящих средних)
    macd = ta.trend.macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_Diff'] = ta.trend.macd_diff(df['Close'])

    # Признак на основе волатильности (True Range Average)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

    # Признаки на основе объема
    df['Volume_SMA_20'] = ta.volume.volume_sma_indicator(df['Volume'], window=20)
    df['Volume_Change'] = df['Volume'].pct_change()

    # Признаки на основе изменения цены
    df['Daily_Return'] = df['Close'].pct_change()
    df['Lag_Return_1'] = df['Daily_Return'].shift(1)
    df['Lag_Return_2'] = df['Daily_Return'].shift(2)

    # Удаляем строки с NaN, которые появились из-за расчета индикаторов
    df = df.dropna()
    print("Признаки сгенерированы. Количество строк после удаления NaN:", len(df))
    return df

# --- 3. Определение целевой переменной ---
def create_target_variable(df, forward_days=1):
    """
    Создает целевую переменную: 1, если цена вырастет на следующий день, 0, если упадет.
    """
    print(f"Создание целевой переменной: предсказание движения цены через {forward_days} день(дней).")
    # Сдвигаем цены закрытия на `forward_days` вперед, чтобы предсказать будущее
    df['Future_Close'] = df['Close'].shift(-forward_days)
    # 1, если будущая цена выше текущей, иначе 0
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    # Удаляем последние `forward_days` строки, так как для них нет будущей цены
    df = df.iloc[:-forward_days]
    print("Целевая переменная создана.")
    return df

# --- 4. Основная логика агента ---
def build_and_train_agent(ticker, start_date, end_date, test_size=0.2, random_state=42):
    """
    Основная функция для построения и обучения AI-агента.
    """
    # 1. Загрузка данных
    data = download_stock_data(ticker, start_date, end_date)

    # 2. Генерация признаков
    data = create_features(data.copy()) # .copy() чтобы избежать SettingWithCopyWarning

    # 3. Определение целевой переменной
    data = create_target_variable(data.copy()) # .copy()

    # Определяем признаки (X) и целевую переменную (y)
    features = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Future_Close', 'Target']]
    X = data[features]
    y = data['Target']

    # Проверка на наличие данных после всех преобразований
    if X.empty or y.empty:
        print("Недостаточно данных для обучения после создания признаков и целевой переменной.")
        return None, None, None, None

    # 4. Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)
    # shuffle=False важен для временных рядов, чтобы не перемешивать хронологию

    print(f"\nРазмеры наборов данных:")
    print(f"Обучающий набор X: {X_train.shape}, y: {y_train.shape}")
    print(f"Тестовый набор X: {X_test.shape}, y: {y_test.shape}")

    # 5. Обучение модели
    print("\nОбучение модели RandomForestClassifier...")
    # Используем RandomForestClassifier, так как он хорошо работает с табличными данными
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced') # balanced для несбалансированных классов
    model.fit(X_train, y_train)
    print("Модель успешно обучена.")

    # 6. Оценка модели
    print("\nОценка производительности модели на тестовом наборе:")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, X_test, y_test, data.loc[X_test.index] # Возвращаем также соответствующую часть исходных данных

# --- 7. Простая стратегия и бэктестинг ---
def backtest_strategy(model, X_test, y_test, historical_data_test):
    """
    Проводит простое бэктестирование торговой стратегии на основе предсказаний модели.
    """
    print("\nЗапуск бэктестинга простой стратегии...")

    predictions = model.predict(X_test)
    
    # Создаем DataFrame для результатов бэктестинга
    strategy_df = pd.DataFrame(index=X_test.index)
    strategy_df['Close'] = historical_data_test['Close']
    strategy_df['Actual_Target'] = y_test
    strategy_df['Predicted_Target'] = predictions

    # Упрощенная стратегия:
    # Покупаем (держим позицию), если модель предсказывает рост (1)
    # Продаем (закрываем позицию/не открываем), если модель предсказывает падение (0)

    # Рассчитываем ежедневные доходности, если бы мы следовали стратегии
    # Если предсказание 1 (рост), то мы "покупаем" и получаем дневную доходность
    # Если предсказание 0 (падение), то мы "не покупаем" и получаем 0 доходности (или не получаем, если шортим)

    # Для простоты: предположим, мы держим позицию, если модель предсказывает рост.
    # Если модель предсказывает падение, мы находимся вне рынка.
    strategy_df['Strategy_Return'] = strategy_df['Close'].pct_change().shift(-1) # Доходность на следующий день
    strategy_df['Strategy_Return'] = strategy_df['Strategy_Return'].fillna(0) # Заполняем NaN нулями

    # Применяем стратегию: только если модель предсказала рост (1), мы получаем доходность
    strategy_df['Strategy_Return_Applied'] = strategy_df['Strategy_Return'] * strategy_df['Predicted_Target']

    # Рассчитываем доходность "купи и держи" для сравнения
    buy_and_hold_returns = strategy_df['Close'].pct_change().fillna(0)

    # Кумулятивная доходность
    strategy_df['Cumulative_Strategy_Return'] = (1 + strategy_df['Strategy_Return_Applied']).cumprod() - 1
    cumulative_buy_and_hold_return = (1 + buy_and_hold_returns).cumprod() - 1

    print(f"\nИтоговая кумулятивная доходность стратегии: {strategy_df['Cumulative_Strategy_Return'].iloc[-1]:.2%}")
    print(f"Итоговая кумулятивная доходность 'купи и держи': {cumulative_buy_and_hold_return.iloc[-1]:.2%}")

    # Визуализация
    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_buy_and_hold_return, label='Купи и держи')
    plt.plot(strategy_df['Cumulative_Strategy_Return'], label='Стратегия AI-агента')
    plt.title(f'Сравнение кумулятивной доходности: {ticker}')
    plt.xlabel('Дата')
    plt.ylabel('Кумулятивная доходность')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return strategy_df

# --- Запуск агента ---
if __name__ == "__main__":
    TICKER = "AAPL"          # Акция, которую мы будем анализировать (например, Apple)
    START_DATE = "2010-01-01"
    END_DATE = "2023-11-01"  # Доступные данные до ноября 2023 г.

    # Строим и обучаем агента
    model, X_test_data, y_test_data, historical_data_for_backtest = build_and_train_agent(TICKER, START_DATE, END_DATE)

    if model is not None:
        # Проводим бэктестинг стратегии
        backtest_results = backtest_strategy(model, X_test_data, y_test_data, historical_data_for_backtest)
    else:
        print("Агент не был построен или обучен из-за недостатка данных.")
