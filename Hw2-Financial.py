import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('cleaned-data.csv')

# Streamlit app
st.title("Data pre-processing and return analysis- Assignment 2")
st.header('Group 1: Saja Tawil- Mahdi Husseini- Mickael Abboud')

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
    
# Select columns containing "Close"
cols = data.columns[data.columns.str.contains('Close')]
close_prices = data[cols]
    
# Resample data to get last closing price of each month
data = close_prices.resample('M').last()
data.reset_index(inplace=True)
    
# Display dataset preview
st.write("### Dataset Preview")
st.dataframe(data)
    
# Calculate simple and log returns
returns = pd.DataFrame()
returns['Date'] = data['Date']
for c in cols:
        name = c.split('_')[1] + '_simple_return'
        name2 = c.split('_')[1] + '_log_return'
        returns[name] = data[c].pct_change()
        returns[name2] = np.log(data[c] / data[c].shift(1))
    
# Display calculated returns
st.write("### Calculated Returns")
st.dataframe(returns.dropna())

# User selection for stock visualization
stock_options = [c.split('_')[1] for c in cols]
selected_stock = st.selectbox("Select a stock to visualize pairwise returns:", stock_options)
    
if selected_stock:
    st.write(f"### Pairwise Plot: {selected_stock} Simple vs Log Return")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(returns[f'{selected_stock}_simple_return'], returns[f'{selected_stock}_log_return'], alpha=0.5)
    ax.set_xlabel(f'{selected_stock} Simple Return')
    ax.set_ylabel(f'{selected_stock} Log Return')
    ax.set_title(f'Pairwise Plot: {selected_stock} Simple vs Log Return')
    ax.grid(True)
    st.pyplot(fig)

# Function to calculate realized volatility
def realized_volatility(x):
    return np.sqrt(np.sum(x**2))

realized_vol = returns.groupby(pd.Grouper(key='Date', freq='M')).apply(
    lambda df: df.filter(like='_log_return').apply(realized_volatility)
)


realized_vol.reset_index(inplace=True)
realized_vol.columns.name = None
realized_vol.index.name = None

# Annualize volatility
annualized_vol = realized_vol.iloc[:, 1:] * np.sqrt(12)
annualized_vol.insert(0, 'Date', realized_vol['Date'])

# Ensure 'Date' is a proper datetime format
realized_vol['Date'] = pd.to_datetime(realized_vol['Date']).dt.date
annualized_vol['Date'] = pd.to_datetime(annualized_vol['Date']).dt.date

realized_vol = realized_vol.sort_values(by='Date')
annualized_vol = annualized_vol.sort_values(by='Date')

# Extract stock columns
stock_columns = [col for col in annualized_vol.columns if col != 'Date']

# Streamlit visualization
st.write("### Realized Volatility and Annualized Volatility")
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot log returns for each stock
for stock in stock_columns:
    ax[0].plot(realized_vol['Date'], realized_vol[stock], label=stock, alpha=0.7)
ax[0].set_title("Log Returns for Stocks")
ax[0].set_ylabel("Log Return")
ax[0].legend(loc="upper right", fontsize=8)

# Plot annualized realized volatility for each stock
for stock in stock_columns:
    ax[1].plot(annualized_vol['Date'], annualized_vol[stock], label=stock, alpha=0.7)
ax[1].set_title("Annualized Realized Volatility")
ax[1].set_ylabel("Volatility")
ax[1].legend(loc="upper right", fontsize=8)

plt.xticks(rotation=45)
plt.xlabel("Date")
plt.tight_layout()
st.pyplot(fig)

st.write("### Around 2021 we can witness the highest volatility. Bank of America has the highest annulaized volatility.")

# Streamlit app title and header
st.title("Stock Returns and Volatility Analysis")
st.header("Interactive Visualization of Log Returns and Volatility")
log_returns = returns.filter(like='_log_return')

log_returns.reset_index(inplace=True)  # If 'Date' is the index, move it to a column
annualized_vol.reset_index(inplace=True)

# Ensure 'Date' exists
if 'Date' not in log_returns.columns:
    log_returns['Date'] = returns['Date']

if 'Date' not in annualized_vol.columns:
    annualized_vol['Date'] = returns['Date']

# Ensure 'Date' is in datetime format
log_returns['Date'] = pd.to_datetime(log_returns['Date'])
annualized_vol['Date'] = pd.to_datetime(annualized_vol['Date'])


# Extract stock columns
stock_columns = [col for col in annualized_vol.columns if col != 'Date']

# User selection for stock visualization
selected_stock = st.selectbox("Select a stock to visualize:", stock_columns)

# Check if the user has made a selection
if selected_stock:
    st.write(f"## {selected_stock} Log Returns and Annualized Realized Volatility")

    # Create the figure and axes
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    
    # Plot log returns
    ax[0].plot(log_returns['Date'], log_returns[selected_stock], label=f"{selected_stock} Log Returns", color='blue', alpha=0.7)
    ax[0].set_title(f"{selected_stock} Log Returns")
    ax[0].set_ylabel("Log Return")
    ax[0].legend()
    ax[0].grid(True)

    # Plot annualized realized volatility
    ax[1].plot(annualized_vol['Date'], annualized_vol[selected_stock], label=f"{selected_stock} Annualized Volatility", color='red', alpha=0.7)
    ax[1].set_title(f"{selected_stock} Annualized Realized Volatility")
    ax[1].set_ylabel("Volatility")
    ax[1].legend()
    ax[1].grid(True)

    # Format x-axis
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Dictionary containing analysis for each stock
stock_analysis = {
    "AAPL": "**AAPL (Apple Inc.)**\n- **Log Returns:** The returns fluctuate significantly over time, with periods of higher volatility visible.\n- **Annualized Volatility:** The realized volatility shows cyclical patterns, with spikes corresponding to market instability.\n- **Interpretation:** AAPL has shown persistent but moderate fluctuations in returns, with volatility spikes that align with market-wide events.",
    
    "ADBE": "**ADBE (Adobe Inc.)**\n- **Log Returns:** Similar to AAPL, ADBE exhibits fluctuations in returns, with some pronounced peaks.\n- **Annualized Volatility:** High spikes in realized volatility at certain periods, indicating market turbulence.\n- **Interpretation:** The returns remain volatile, and the realized volatility suggests that ADBE experiences occasional high-risk periods.",
    
    "AMZN": "**AMZN (Amazon Inc.)**\n- **Log Returns:** Larger fluctuations in returns compared to AAPL and ADBE.\n- **Annualized Volatility:** Several pronounced spikes, particularly around 2016 and 2020.\n- **Interpretation:** AMZN has experienced periods of high risk, particularly during major financial or company-specific events.",
    
    "BA": "**BA (Boeing Co.)**\n- **Log Returns:** Generally stable with some sharp movements.\n- **Annualized Volatility:** A significant spike around 2020, likely due to the impact of COVID-19.\n- **Interpretation:** The volatility and returns show major fluctuations, reflecting sensitivity to industry events like the pandemic.",
    
    "BAC": "**BAC (Bank of America)**\n- **Log Returns:** Moderate fluctuations in returns.\n- **Annualized Volatility:** Notable spikes around 2020 and other financial crises.\n- **Interpretation:** BAC exhibits high sensitivity to financial crises, with volatility aligning with macroeconomic downturns.",
    
    "BLK": "**BLK (BlackRock Inc.)**\n- **Log Returns:** Fairly stable with some periodic increases.\n- **Annualized Volatility:** A few spikes but overall relatively controlled.\n- **Interpretation:** BLK maintains a relatively stable return profile, with controlled volatility except during major economic shifts.",
    
    "CSCO": "**CSCO (Cisco Systems Inc.)**\n- **Log Returns:** Periodic sharp changes but mostly stable.\n- **Annualized Volatility:** Spikes in volatility align with downturns in the tech sector.\n- **Interpretation:** CSCO is less volatile than other tech stocks but still experiences significant periods of risk.",
    
    "CVS": "**CVS (CVS Health Corp.)**\n- **Log Returns:** Noticeable fluctuations, with some periods of strong movement.\n- **Annualized Volatility:** A recent spike in volatility, likely reflecting market conditions.\n- **Interpretation:** CVS has become more volatile in recent years, indicating increased uncertainty in the healthcare sector.",
    
    "DAL": "**DAL (Delta Air Lines)**\n- **Log Returns:** Significant drops and recoveries, with large swings.\n- **Annualized Volatility:** A massive spike in 2020 due to the pandemic.\n- **Interpretation:** DAL is highly susceptible to external shocks, particularly travel restrictions and fuel price changes.",
    
    "DPZ": "**DPZ (Domino's Pizza)**\n- **Log Returns:** Consistently fluctuating with some strong movements.\n- **Annualized Volatility:** Noticeable volatility peaks, especially around market downturns.\n- **Interpretation:** DPZ remains moderately volatile but shows resilience compared to other sectors."
}

# Fixed Conclusion
conclusion = """
## **Conclusion**
- **High Volatility Stocks:** BA, DAL, AMZN, ADBE, BAC – These stocks experience sharp spikes in volatility, mostly due to sector-wide crises or macroeconomic changes.
- **Moderate Volatility Stocks:** AAPL, CSCO, DPZ – These stocks fluctuate, but with fewer extreme volatility spikes.
- **Stable Stocks:** BLK, CVS – These stocks exhibit relatively stable log returns and fewer periods of excessive volatility.

**Overall, the relationship between log returns and volatility suggests that higher return fluctuations generally coincide with increased realized volatility, which is more prominent in sectors affected by macroeconomic crises, such as airlines, finance, and tech.**
"""

# Streamlit UI
selected_stock = st.selectbox("Select a stock to view analysis:", list(stock_analysis.keys()))

# Display stock-specific analysis
st.write("### Analysis of Log Returns and Annualized Realized Volatility")
st.markdown(stock_analysis[selected_stock])

# Display fixed conclusion after analysis
st.write(conclusion)

# Title
st.title("📈 Inflation-Adjusted Stock Returns Analysis")

# Load CPI Data
st.subheader("Loading Consumer Price Index (CPI) Data")
cpi = pd.read_csv('cpi2.csv')

# Convert TIME_PERIOD to datetime and sort data
cpi['TIME_PERIOD'] = pd.to_datetime(cpi['TIME_PERIOD'])
cpi.sort_values(by='TIME_PERIOD', inplace=True)
cpi.reset_index(drop=True, inplace=True)

# Calculate monthly inflation rate
cpi['Inflation'] = cpi['OBS_VALUE'].pct_change()

# Display CPI with inflation rate
st.write("### Inflation Data")
st.dataframe(cpi[['TIME_PERIOD', 'Inflation']].dropna().head(10))

# Load Returns Data
st.subheader("Loading Stock Returns Data")
returns['Date'] = pd.to_datetime(returns['Date'])

# Adjust date format for merging (Month & Year only)
returns['YearMonth'] = returns['Date'].dt.to_period('M').astype(str)
cpi['YearMonth'] = cpi['TIME_PERIOD'].dt.to_period('M').astype(str)

# Merge inflation rate into returns data
returns = returns.merge(cpi[['YearMonth', 'Inflation']], on='YearMonth', how='left')

# Adjust returns for inflation
for col in returns.columns:
    if '_simple_return' in col:
        new_col = col.replace('_simple_return', '_real_return')
        returns[new_col] = (1 + returns[col]) / (1 + returns['Inflation']) - 1

    if '_log_return' in col:
        new_col = col.replace('_log_return', '_real_log_return')
        returns[new_col] = returns[col] - returns['Inflation']

# Handle missing values using backward-fill
returns.fillna(method='bfill', inplace=True)

# Display results
st.subheader("📊 Inflation-Adjusted Returns")
st.write("Below is the adjusted stock return data after accounting for inflation:")
st.dataframe(returns.head(10))

st.write("""
### 📌 How We Adjusted for Inflation:
- **For Simple Returns:** We used the formula  
  \[
  (1 + r) / (1 + \text{inflation}) - 1
  \]
- **For Log Returns:** We applied simple subtraction  
  \[
  \text{log return} - \text{inflation}
  \]
- **Why?** Log differences approximate relative changes, making them directly comparable.

""")

# Extract stock columns
stock_columns = [col.replace('_simple_return', '') for col in returns.columns if '_simple_return' in col]

# Streamlit UI
st.title("Stock Returns: Nominal vs. Inflation-Adjusted")
st.write("Select a stock to visualize its nominal and inflation-adjusted returns over time.")

# Dropdown for stock selection
selected_stock = st.selectbox("Choose a Stock:", stock_columns)

# Plot the selected stock's returns
fig, ax = plt.subplots(figsize=(10, 5))

# Plot nominal simple return
ax.plot(returns['Date'], returns[f'{selected_stock}_simple_return'], label=f'{selected_stock} Nominal Return', color='blue', alpha=0.6)

# Plot real (inflation-adjusted) simple return
ax.plot(returns['Date'], returns[f'{selected_stock}_real_return'], label=f'{selected_stock} Inflation-Adjusted Return', color='red', alpha=0.6)

ax.set_xlabel("Date")
ax.set_ylabel("Return")
ax.set_title(f"{selected_stock} Returns: Nominal vs. Inflation-Adjusted")
ax.legend()

# Show the plot in Streamlit
st.pyplot(fig)

st.title("Stock Log Returns: Nominal vs. Inflation-Adjusted")
# Display the message to guide the user
st.write("Select a stock to visualize its nominal and inflation-adjusted log returns over time.")

# User selection for stock visualization with a unique key
stock_options = [c.split('_')[1] for c in cols]
selected_stock = st.selectbox("Select a stock to visualize its nominal and inflation-adjusted log returns over time:", stock_options, key="stock_selectbox")

if selected_stock:
    # Visualization code goes here...
    st.write(f"### {selected_stock} Nominal vs. Inflation-Adjusted Log Returns")
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot nominal log return
    ax.plot(returns['Date'], returns[f'{selected_stock}_log_return'], label=f'{selected_stock} Nominal Log Return', color='blue', alpha=0.6)

    # Plot real (inflation-adjusted) log return
    ax.plot(returns['Date'], returns[f'{selected_stock}_real_log_return'], label=f'{selected_stock} Inflation-Adjusted Log Return', color='red', alpha=0.6)

    ax.set_xlabel('Date')
    ax.set_ylabel('Return')
    ax.set_title(f'{selected_stock} Returns: Nominal vs. Inflation-Adjusted')
    ax.legend()
    st.pyplot(fig)

# Add the analysis as markdown
st.markdown("""
### **Analysis of Nominal vs. Inflation-Adjusted Returns**

We didn't see much of a change for the two plots for each stock, and that's because the inflation rate is small, and integrating it in the formula doesn't add a major observable difference when it comes to visualizing data, except in high-inflation periods. Especially since we are dealing with the US market, where inflation is very unlikely to have occurred during the period 2014 - 2024.

The nominal and inflation-adjusted return plots show minimal differences for each stock over the period 2014–2024. This is because the US inflation rate remained relatively low and stable for most of this time.

The inflation-adjusted return formula accounts for price level changes, but since inflation was modest, its impact on stock returns was negligible in most cases.

Noticeable differences appeared in high-inflation periods, particularly around 2021–2022, when inflation spiked due to global economic factors (e.g., supply chain disruptions, stimulus-driven demand).

Since the US has historically low inflation volatility, stock returns tend to remain largely unaffected except during inflationary shocks. The relationship between simple and log returns remains consistent, further confirming that inflation had only a minor impact except in specific high-inflation periods.

### **Conclusion**
Inflation is an important factor in long-term investment returns, but in a low-inflation economy like the US (during most of 2014–2024), its short-term impact on stock returns is limited.

Only in high-inflation periods (e.g., 2021–2022) do we observe a more noticeable gap between nominal and real returns, highlighting how inflation erodes purchasing power.

These findings reinforce the idea that investors in developed markets should monitor inflation trends but may not see major inflation-driven distortions in short- to mid-term stock returns.
""")


btc = pd.read_csv('BTCUSDT_trades.csv')
xrp = pd.read_csv('XRPUSDT_trades.csv')
doge = pd.read_csv('DOGEUSDT_trades.csv')

dfs = [btc, xrp, doge]
dfs_names = ['BTCUSDT', 'XRPUSDT', 'DOGEUSDT']

st.title("Cryptocurrency trade data")
# Display BTC data
st.subheader("BTCUSDT Trades Data")
st.dataframe(btc.head())

# Display XRP data
st.subheader("XRPUSDT Trades Data")
st.dataframe(xrp.head())

# Display DOGE data
st.subheader("DOGEUSDT Trades Data")
st.dataframe(doge.head())

processed_dfs = []

for i, df in enumerate(dfs):
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['quoteQty'] = pd.to_numeric(df['quoteQty'], errors='coerce')

    try:
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    except Exception:
        df['time'] = pd.to_datetime(df['time'])
    
    processed_dfs.append(df)


# Function to compute OHLC, VWAP, volume, and count
def get_bars(df_grouped, add_time=False):
    # Compute OHLC for the 'price' column
    ohlc = df_grouped['price'].ohlc()
    # Compute VWAP (Volume-Weighted Average Price)
    vwap = df_grouped.apply(lambda x: np.average(x['price'], weights=x['qty'])).to_frame("vwap")
    # Total volume and count (number of trades)
    vol = df_grouped['qty'].sum().to_frame("vol")
    cnt = df_grouped['qty'].count().to_frame("cnt")
    
    if add_time:
        # Get the last time of each group (index is now datetime)
        t = df_grouped['time'].last().to_frame("time")
        res = pd.concat([t, ohlc, vwap, vol, cnt], axis=1)
    else:
        res = pd.concat([ohlc, vwap, vol, cnt], axis=1)
    
    return res

# Load CSV files
dfs = {
    "BTCUSDT": pd.read_csv('BTCUSDT_trades.csv'),
    "XRPUSDT": pd.read_csv('XRPUSDT_trades.csv'),
    "DOGEUSDT": pd.read_csv('DOGEUSDT_trades.csv')
}

# Convert time column to datetime
for name, df in dfs.items():
    try:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Automatically detects format
    except Exception as e:
        st.error(f"Error converting time column for {name}: {e}")

    df.set_index("time", inplace=True)  # Set time as index for grouping

# Streamlit UI
st.title("Cryptocurrency Time Bars")

# Select which cryptocurrency to display
selected_crypto = st.selectbox("Select a cryptocurrency:", list(dfs.keys()))

# Process the selected cryptocurrency
df = dfs[selected_crypto]
df_grouped_time = df.groupby(pd.Grouper(freq="1Min"))
bars = get_bars(df_grouped_time).reset_index(drop=True)

# Display results
st.subheader(f"Time Bars for {selected_crypto}")
st.dataframe(bars)


st.write("BTCUSDT: Shows a dominant bearish candle, suggesting a major price movement in a single interval with little fluctuation afterward.")
st.write("XRPUSDT: Displays a steady downtrend with significant price drops and intraday volatility.")
st.write("DOGEUSDT: Has the most fluctuations, showing alternating bullish and bearish movements, suggesting higher volatility.")

st.write("Time bars offer a structured way to analyze price trends over fixed intervals. However, they may not always capture rapid market movements efficiently, making them less responsive than tick, volume, or dollar bars in high-volatility scenarios.")


