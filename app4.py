import streamlit as st
import os
from datetime import datetime
import pandas as pd
import yfinance as yf

from tefas import Crawler
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings

###############################################################################
# 1. GLOBAL SETUP & CACHING
###############################################################################

# Avoid certain warnings
warnings.simplefilter("ignore", UserWarning)

# Set Streamlit page config
st.set_page_config(
    page_title="YatırıMekko",
    layout="wide"
)

tefas = Crawler()

# Cache TEFAS mutual fund data for today's date
@st.cache_data(ttl=3600)
def fetch_tefas_data():
    today_date = datetime.today().strftime('%Y-%m-%d')
    return tefas.fetch(
        start=today_date,
        columns=["code", "date", "price", "title", "stock"]
    )

tefas_data = fetch_tefas_data()

# BIST stocks
@st.cache_data(ttl=360000)
def get_bist_stocks():
    url = "https://finans.mynet.com/borsa/hisseler/"
    response = requests.get(url)
    if response.status_code != 200:
        st.warning("Failed to retrieve BIST stocks.")
        return pd.DataFrame(columns=["Yahoo Finance Code", "Simplified Code"])  # empty

    soup = BeautifulSoup(response.text, 'html.parser')
    stock_elements = soup.select("table tr td a")
    bist_stocks = [
        stock.text.strip().split(" ")[0] + ".IS"
        for stock in stock_elements
        if stock.text.strip()
    ]

    # Create DataFrame with two columns
    bist_df = pd.DataFrame({
        "Simplified Code": [s.split(".")[0] for s in bist_stocks],  # remove '.IS'
        "Yahoo Finance Code": bist_stocks
    })
    return bist_df

bist_stock_df = get_bist_stocks()

@st.cache_data(ttl=60)
def scrape_doviz_emtia():
    """
    Scrapes Gram-based commodity prices (e.g., Gram Altın, Gram Gümüş) from doviz.com/emtia.
    Returns a dict of { 'GRAM ALTIN': price, 'GRAM GÜMÜŞ': price, ... }
    """
    url = "https://www.doviz.com/emtia"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                       " AppleWebKit/537.36 (KHTML, like Gecko)"
                       " Chrome/91.0.4472.124 Safari/537.36")
    }
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        st.warning(f"Failed to retrieve emtia data: {response.status_code}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")
    commodity_items = soup.find_all("div", class_="item")

    emtia_dict = {}
    for item in commodity_items:
        name_element = item.find("span", class_="name")
        price_element = item.find("span", class_="value")
        if name_element and price_element:
            name = name_element.text.strip()
            price = price_element.text.strip()
            if "GRAM" in name.upper():
                cleaned_price = price.replace(".", "").replace(",", ".")
                emtia_dict[name.upper()] = float(cleaned_price)
    return emtia_dict

emtia_prices_dict = scrape_doviz_emtia()

@st.cache_data(ttl=60)
def fetch_exchange_rates():
    """Fetch daily close values of currencies from Yahoo (USDTRY=X, EURTRY=X, etc.)."""
    currency_mapping = {
        "USD": "USDTRY=X",
        "EUR": "EURTRY=X",
        "JPY": "JPYTRY=X",
        "GBP": "GBPTRY=X",
        "AED": "AEDTRY=X",
        "CAD": "CADTRY=X",
        "CHF": "CHFTRY=X"
    }
    exchange_rates = {}
    for _, ticker in currency_mapping.items():
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                exchange_rates[ticker] = data["Close"].iloc[-1]
            else:
                exchange_rates[ticker] = None
        except Exception:
            exchange_rates[ticker] = None
    return exchange_rates

exchange_rates = fetch_exchange_rates()

# Additional dictionaries from your original script
currency_mapping = {
    "USD": "USDTRY=X",
    "EUR": "EURTRY=X",
    "JPY": "JPYTRY=X",
    "GBP": "GBPTRY=X",
    "AED": "AEDTRY=X",
    "CAD": "CADTRY=X",
    "CHF": "CHFTRY=X"
}

emtia_mapping1 = {
    "GramAltın": "GRAM ALTIN",
    "GRAMALTIN": "GRAM ALTIN",
    "gramaltin": "GRAM ALTIN",
}

emtia_mapping2 = {
    "GramGümüş": "GRAM GÜMÜŞ",
    "GRAMGÜMÜŞ": "GRAM GÜMÜŞ",
    "gramgumus": "GRAM GÜMÜŞ"
}

crypto_df = pd.DataFrame({
    "User input": ["BTC", "ETH", "USDT", "XRP", "BNB", "SOL", "USDC", "DOGE", "ADA",
                   "STETH", "WTRX", "TRX", "LINK", "XLM", "AVAX", "FDUSD", "LTC"],
    "Symbol": ["BTC-USD", "ETH-USD", "USDT-USD", "XRP-USD", "BNB-USD", "SOL-USD",
               "USDC-USD", "DOGE-USD", "ADA-USD", "STETH-USD", "WTRX-USD", "TRX-USD",
               "LINK-USD", "XLM-USD", "AVAX-USD", "FDUSD-USD", "LTC-USD"]
})

###############################################################################
# 2. CORE FUNCTIONS ADAPTED FOR STREAMLIT
###############################################################################

def parse_user_holdings(portfolio_text: str):
    """
    Parses multiline text from the user. Each line should be "CODE QUANTITY".
    Example lines:
      USD 1000
      TUPRS 50
      AAPL 10
    Returns a list of holdings: [ { code: ..., quantity: ..., category: ... }, ... ]
    """
    lines = portfolio_text.strip().split("\n")
    holdings = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            code, quantity_str = line.split()
            code = code.upper()
            category = "Unknown"

            #EUROBOND Handling
            if code == "EUROBOND":
                category = "Eurobond"
            elif code == "TRY":
                category = "Türk Lirası"  # Mark as cash
            elif code in currency_mapping:
                code = currency_mapping[code]
                category = "Döviz"
            elif code in emtia_mapping1:
                code = emtia_mapping1[code]
                category = "Altın"
            elif code in emtia_mapping2:
                code = emtia_mapping2[code]
                category = "Gümüş"
            # Convert BIST stock codes
            elif code in bist_stock_df["Simplified Code"].values:
                code = bist_stock_df.loc[bist_stock_df["Simplified Code"] == code, "Yahoo Finance Code"].values[0]
                category = "TR hisse"
            # Convert Crypto codes
            elif code in crypto_df["User input"].values:
                symbol = crypto_df.loc[crypto_df["User input"] == code, "Symbol"].values[0]
                code = symbol
                category = "Kripto"

            holdings.append({
                "code": code,
                "quantity": float(quantity_str),
                "category": category
            })
        except ValueError:
            st.warning(f"Skipping invalid line: '{line}' (must be CODE QUANTITY)")
    return holdings

def classify_assets(holdings, tefas_df):
    """
    Classifies each holding as 'Döviz', 'Kripto', 'Altın', 'Gümüş',
    or determines if it's a TEFAS mutual fund, a stock, an ETF, etc.
    """
    for holding in holdings:
        code = holding["code"]
        category = holding["category"]

        # Already assigned categories - skip re-check
        if category in ["Döviz", "Kripto", "Altın", "Gümüş","Türk Lirası", "Eurobond"]:
            continue

        # Check if it's a TEFAS mutual fund
        fund_info = tefas_df[tefas_df["code"] == code]
        if not fund_info.empty:
            fund_stock_ratio = fund_info.iloc[-1]["stock"]
            fund_title = fund_info.iloc[-1]["title"].lower()
            if "borçlanma" in fund_title or "bond" in fund_title:
                holding["category"] = "Borçlanma araçları fonu"
            elif "para pi̇yasasi" in fund_title:
                holding["category"] = "Para piyasası fonu"
            elif fund_stock_ratio > 50:
                holding["category"] = "TR hisse fonu"
            else:
                holding["category"] = "Diğer fon"
            continue

        # Otherwise, let's see if it’s an ETF/Stock from Yahoo
        try:
            ticker = yf.Ticker(code)
            qtype = ticker.info.get("quoteType", "Unknown").upper()
            market = ticker.info.get("fullExchangeName", "").lower()
            if qtype == "ETF":
                holding["category"] = "Yabancı ETF"
            elif qtype == "EQUITY":
                if "istanbul" in market:
                    holding["category"] = "TR hisse"
                else:
                    holding["category"] = "Yabancı hisse"
            else:
                holding["category"] = "Unknown"
        except Exception:
            # If we can't fetch anything, just keep it as "Unknown"
            pass

    return holdings

def fetch_prices(holdings, tefas_df):
    """Fetch the last known price in TRY for each holding."""
    prices = {}
    for h in holdings:
        code, category = h["code"], h["category"]

        # 1. Döviz => use exchange_rates
        if category == "Döviz":
            prices[code] = exchange_rates.get(code, None)
            continue

        # 2. Altın / Gümüş => use emtia_prices_dict
        if category in ["Altın", "Gümüş"]:
            # code is something like "GRAM ALTIN" or "GRAM GÜMÜŞ" in uppercase
            # e.g. stored as "GRAM ALTIN" in your dictionary
            # your dict keys are uppercase => "GRAM ALTIN", "GRAM GÜMÜŞ"
            prices[code] = emtia_prices_dict.get(code, None)
            continue

        # 3. TEFAS mutual funds => tefas_df
        if category in ["TR hisse fonu", "Borçlanma araçları fonu", "Para piyasası fonu", "Diğer fon"]:
            fund_data = tefas_df[tefas_df["code"] == code]
            if not fund_data.empty:
                prices[code] = fund_data.iloc[-1]["price"]
            else:
                prices[code] = None
            continue

        # 4. Stocks, ETFs, Kripto => Yahoo Finance close. Convert to TRY if needed.
        if category in ["Yabancı ETF", "TR hisse", "Yabancı hisse", "Kripto", "Unknown"]:
            try:
                ticker = yf.Ticker(code)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    last_price = hist["Close"].iloc[-1]
                    # Currency might be "USD" => we multiply by USDTRY
                    yahoo_currency = ticker.info.get("currency", "TRY")  # e.g. "USD" or "TRY"
                    if yahoo_currency.upper() != "TRY":
                        # Compose something like "USDTRY=X"
                        pair = (yahoo_currency + "TRY=X").upper()
                        fx_rate = exchange_rates.get(pair, None)
                        if fx_rate is not None:
                            last_price *= fx_rate
                    prices[code] = last_price
                else:
                    prices[code] = None
            except Exception as e:
                st.error(f"Error fetching price for {code}: {e}")
                prices[code] = None

        if category in ["Türk Lirası"]:
            prices[code] = 1
            
        if category in ["Eurobond"]:
            prices[code] = 1000 * exchange_rates.get("USDTRY=X", None)
            
    return prices

def calculate_net_worth(holdings, prices):
    total = 0
    usdtry_rate = exchange_rates.get("USDTRY=X", 1)  # Get USD/TRY rate

    for h in holdings:
        code, qty, cat = h["code"], h["quantity"], h["category"]

        # Special handling for Eurobond
        if cat == "Eurobond":
            total += qty * 1000 * usdtry_rate  # Eurobonds are in USD, so we convert

        # Turkish Lira (TRY) Cash
        elif cat == "Nakit":
            total += qty  # TRY is already in TRY

        # Regular assets
        else:
            price = prices.get(code, 0)
            if price is not None:
                total += price * qty

    return total

def display_portfolio_table(holdings, prices):
    """
    Displays the portfolio table with a % share column.
    - Price and Quantity: 2 decimal places.
    - Total Value and % Share: No decimals.
    - Sorted from largest to smallest position.
    """
    df = pd.DataFrame(holdings)
    df["Price (TRY)"] = df["code"].map(prices)
    df["Total Value (TRY)"] = df["quantity"] * df["Price (TRY)"]

    # Drop rows where price is NaN
    df = df.dropna(subset=["Price (TRY)"])

    # Compute % Share
    total_value = df["Total Value (TRY)"].sum()
    df["% Share"] = (df["Total Value (TRY)"] / total_value) * 100

    # Format numerical values
    df["Price (TRY)"] = df["Price (TRY)"].round(2)  # 2 decimal places, thousands separator
    df["Total Value (TRY)"] = df["Total Value (TRY)"].astype(int)  # No decimals, thousands separator
    df["% Share"] = df["% Share"].round(1)  # No decimals

    # Sort positions from largest to smallest
    df = df.sort_values(by="% Share", ascending=False)

    return df

# Asset Classes Mapping
asset_classes = {
    "Borsa": ["Yabancı ETF", "TR hisse fonu", "TR hisse", "Yabancı hisse"],
    "Tahvil": ["Diğer fon", "Borçlanma araçları fonu","Eurobond"],
    "Nakit": ["Para piyasası fonu", "Döviz","Türk Lirası"],
    "Emtia & Kripto": ["Kripto", "Altın", "Gümüş"]
}

def categorize_holdings_for_mekko(holdings, prices):
    """Group holdings by major asset class for the Mekko chart."""
    data = {"Borsa": {}, "Tahvil": {}, "Nakit": {}, "Emtia & Kripto": {}}
    for h in holdings:
        cat = h["category"]
        val = prices.get(h["code"], 0) * h["quantity"]
        for class_name, categories in asset_classes.items():
            if cat in categories:
                if cat not in data[class_name]:
                    data[class_name][cat] = 0
                data[class_name][cat] += val
                break
    return data

def create_mekko_chart(investments):
    """
    Creates the Mekko chart with total displayed in both TRY and USD.
    """
    asset_class_totals = {asset: sum(sub.values()) for asset, sub in investments.items()}
    asset_class_totals = {k: v for k, v in asset_class_totals.items() if v > 0}

    total_investment_try = sum(asset_class_totals.values())
    usdtry_rate = exchange_rates.get("USDTRY=X", 1)
    total_investment_usd = total_investment_try / usdtry_rate  # Convert to USD

    if total_investment_try <= 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No investment data to plot.", ha="center", va="center")
        return fig, ax

    sorted_assets = sorted(asset_class_totals.items(), key=lambda x: x[1], reverse=True)
    sorted_investments = {asset: dict(sorted(investments[asset].items(), key=lambda x: x[1], reverse=True)) for asset, _ in sorted_assets}

    fig, ax = plt.subplots(figsize=(10, 6))
    x_start = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_assets)))

    for (asset_class, sub_dict), color in zip(sorted_investments.items(), colors):
        width = (asset_class_totals[asset_class] / total_investment_try) * 100
        y_start = 0

        for sub_class, amount in sub_dict.items():
            height = (amount / asset_class_totals[asset_class]) * 100 if asset_class_totals[asset_class] else 0
            rect = plt.Rectangle((x_start, y_start), width, height, color=color, edgecolor='white', linewidth=3)
            ax.add_patch(rect)

            sub_share = (amount / total_investment_try) * 100 if total_investment_try > 0 else 0
            ax.text(x_start + width / 2, y_start + height / 2,
                    f"{sub_class}\n({sub_share:.1f}%)",
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

            if y_start > 0:
                ax.plot([x_start, x_start + width], [y_start, y_start], color='white', linewidth=3)

            y_start += height

        rect_border = plt.Rectangle((x_start, 0), width, 100, edgecolor='white', linewidth=4, fill=False)
        ax.add_patch(rect_border)

        asset_share = (asset_class_totals[asset_class] / total_investment_try) * 100
        ax.text(x_start + width / 2, -5, asset_class,
                ha='center', va='top', fontsize=10, color='black', fontweight='bold')
        ax.text(x_start + width / 2, 100, f"{asset_share:.1f}%",
                ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

        x_start += width

    ax.set_xlim(0, 100)
    ax.set_ylim(-10, 120)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Varlık Sınıflarına Göre Portföy Dağılımı", fontweight='bold')

    # Show total in both TRY and USD
    plt.text(105, 120, f"Total: {total_investment_try:,.0f} TRY\n(~{total_investment_usd:,.0f} USD)",
             ha='right', va='top', fontsize=12, fontweight='bold', color='black')

    return fig, ax




###############################################################################
# 3. STREAMLIT APP LAYOUT
###############################################################################

def main():
    st.title("YatırıMekko: Portföyünü tek yerden takip et! Gerçekten.")
    st.markdown(
        """
        YatırıMekko **TEFAS fonları, BIST hisseleri, Yabancı piyasalardaki hisse ve ETF'leri, Emtia'ları (Altın, Gümüş), Eurobond'u,
        Nakit paranı ve Kriptoparaları** destekler. 
        
        **Portföyünü soldaki kutuya aşağıdaki formatta satır satır gir:**
        
        **ENSTRÜMANKODU  ADET**
        
    **Örnekler:**

    - **TEFAS Fonları:** `NNF 2000`  
    - **BIST Hisseleri:** `THYAO 200`  
    - **Yabancı Hisseler:** `AAPL 10`  
    - **Yabancı ETF'ler:** `QQQM 5`  
    - **Emtia'lar:** `GRAMALTIN 10`  
    - **Eurobond:** `EUROBOND 8`  
    - **Nakit:** `USD 1000` • `TRY 50000`  
    - **Kriptoparalar:** `BTC 0.1`
    """,
    unsafe_allow_html=True

    Bitirdikten sonra **Hesapla** tuşuna bas ve portföy dağılımını incele.
    """
    )

    with st.sidebar:
        st.header("Portföyünü Gir")
        portfolio_text = st.text_area(
            "Format: KOD ADET",
            height=200
        )
        calc_button = st.button("Hesapla")


    # ------------------  User clicked Calculate  ---------------------
    if not portfolio_text.strip():
        st.warning("No portfolio data provided. Please enter something in the sidebar.")
        return

    # 1. Parse user holdings
    user_holdings = parse_user_holdings(portfolio_text)

    if not user_holdings:
        st.warning("No valid portfolio lines were found. Please check your format.")
        return

    # 2. Classify assets
    classified_holdings = classify_assets(user_holdings, tefas_data)

    # 3. Fetch prices
    prices = fetch_prices(classified_holdings, tefas_data)

    # 4. Calculate net worth
    net_worth = calculate_net_worth(classified_holdings, prices)

    # 5. Display portfolio table
    st.subheader("Tablo Görünümü")
    portfolio_df = display_portfolio_table(classified_holdings, prices)
    # Show table
    st.data_editor(portfolio_df, use_container_width=True,hide_index=True,column_config={"% Share": st.column_config.NumberColumn("% Share", format="%.1f%%")})  # Display as percentage with 1 decimal

    # 6. Categorize holdings for Mekko
    investments = categorize_holdings_for_mekko(classified_holdings, prices)

    # 7. Create & display Mekko chart
    st.subheader("Mekko Görünümü")
    fig, ax = create_mekko_chart(investments)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
