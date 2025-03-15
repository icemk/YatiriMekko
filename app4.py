import streamlit as st

st.set_page_config(page_title="YatırıMekko", layout="wide")

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from tefas import Crawler
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import warnings
from supabase import create_client, Client
import uuid
from streamlit_cookies_manager import EncryptedCookieManager



# These must be stored in your Streamlit secrets under the [SUPABASE] section
SUPABASE_URL = st.secrets["SUPABASE"]["URL"]
SUPABASE_KEY = st.secrets["SUPABASE"]["KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

###############################################################################
# 1. GLOBAL SETUP & CACHING
###############################################################################

warnings.simplefilter("ignore", UserWarning)


tefas = Crawler()

# We define these variables here, but set them to None until we call get_data().
tefas_data = None
bist_stock_df = None
emtia_prices_dict = None
exchange_rates = None


def retrieve_data(user_id):
    try:
        response = supabase.table("yatirimekko").select("*").eq("user_id", user_id).execute()
        if response.data:
            return response.data[0]  # Assuming one record per user
        else:
            return None
    except Exception as e:
        st.error(f"Error retrieving data: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_tefas_data():
    def is_valid_tefas_data(date_str):
        try:
            data = tefas.fetch(start=date_str, columns=["code", "date", "price", "title", "stock"])
            return not data.empty
        except:
            return False

    days_back = 0
    while True:
        check_date = (datetime.today() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        if is_valid_tefas_data(check_date):
            return tefas.fetch(start=check_date, columns=["code", "date", "price", "title", "stock"])
        days_back += 1

@st.cache_data(ttl=360000)
def get_bist_stocks():
    url = "https://finans.mynet.com/borsa/hisseler/"
    response = requests.get(url)
    if response.status_code != 200:
        st.warning("Failed to retrieve BIST stocks.")
        return pd.DataFrame(columns=["Yahoo Finance Code", "Simplified Code"])

    soup = BeautifulSoup(response.text, 'html.parser')
    stock_elements = soup.select("table tr td a")
    bist_stocks = [
        stock.text.strip().split(" ")[0] + ".IS"
        for stock in stock_elements
        if stock.text.strip()
    ]

    bist_df = pd.DataFrame({
        "Simplified Code": [s.split(".")[0] for s in bist_stocks],
        "Yahoo Finance Code": bist_stocks
    })
    return bist_df

@st.cache_data(ttl=60)
def scrape_doviz_emtia():
    url = "https://www.doviz.com/emtia"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/91.0.4472.124 Safari/537.36"
        )
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

@st.cache_data(ttl=60)
def fetch_exchange_rates():
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

###############################################################################
# Define a simple function that sets all global data variables in one shot
###############################################################################
def get_data():
    global tefas_data, bist_stock_df, emtia_prices_dict, exchange_rates
    tefas_data = fetch_tefas_data()
    bist_stock_df = get_bist_stocks()
    emtia_prices_dict = scrape_doviz_emtia()
    exchange_rates = fetch_exchange_rates()


###############################################################################
# 2. CORE FUNCTIONS ADAPTED FOR STREAMLIT
###############################################################################

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

def parse_user_holdings(portfolio_text: str):
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

            if code == "EUROBOND":
                category = "Eurobond"
            elif code == "TRY":
                category = "Türk Lirası"
            elif code in currency_mapping:
                code = currency_mapping[code]
                category = "Döviz"
            elif code in emtia_mapping1:
                code = emtia_mapping1[code]
                category = "Altın"
            elif code in emtia_mapping2:
                code = emtia_mapping2[code]
                category = "Gümüş"
            elif code in bist_stock_df["Simplified Code"].values:
                code = bist_stock_df.loc[bist_stock_df["Simplified Code"] == code, "Yahoo Finance Code"].values[0]
                category = "TR hisse"
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
    for holding in holdings:
        code = holding["code"]
        category = holding["category"]

        if category in ["Döviz", "Kripto", "Altın", "Gümüş", "Türk Lirası", "Eurobond"]:
            continue

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
            pass

    return holdings

def fetch_prices(holdings, tefas_df):
    prices = {}
    for h in holdings:
        code, category = h["code"], h["category"]

        if category == "Döviz":
            prices[code] = exchange_rates.get(code, None)
            continue

        if category in ["Altın", "Gümüş"]:
            prices[code] = emtia_prices_dict.get(code, None)
            continue

        if category in ["TR hisse fonu", "Borçlanma araçları fonu", "Para piyasası fonu", "Diğer fon"]:
            fund_data = tefas_df[tefas_df["code"] == code]
            if not fund_data.empty:
                prices[code] = fund_data.iloc[-1]["price"]
            else:
                prices[code] = None
            continue

        if category in ["Yabancı ETF", "TR hisse", "Yabancı hisse", "Kripto", "Unknown"]:
            try:
                ticker = yf.Ticker(code)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    last_price = hist["Close"].iloc[-1]
                    yahoo_currency = ticker.info.get("currency", "TRY")
                    if yahoo_currency.upper() != "TRY":
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

        if category == "Türk Lirası":
            prices[code] = 1

        if category == "Eurobond":
            prices[code] = 1000 * exchange_rates.get("USDTRY=X", None)

    return prices

def calculate_net_worth(holdings, prices):
    total = 0
    usdtry_rate = exchange_rates.get("USDTRY=X", 1)

    for h in holdings:
        code, qty, cat = h["code"], h["quantity"], h["category"]

        if cat == "Eurobond":
            total += qty * 1000 * usdtry_rate
        else:
            price = prices.get(code, 0)
            if price is not None:
                total += price * qty

    return total

def display_portfolio_table(holdings, prices):
    df = pd.DataFrame(holdings)
    df["Price (TRY)"] = df["code"].map(prices)
    df["Total Value (TRY)"] = df["quantity"] * df["Price (TRY)"]

    df = df.dropna(subset=["Price (TRY)"])

    total_value = df["Total Value (TRY)"].sum()
    df["% Share"] = (df["Total Value (TRY)"] / total_value) * 100

    df["Price (TRY)"] = df["Price (TRY)"].round(2)
    df["Total Value (TRY)"] = df["Total Value (TRY)"].astype(int)
    df["% Share"] = df["% Share"].round(1)

    df = df.sort_values(by="% Share", ascending=False)
    return df

asset_classes = {
    "Borsa": ["Yabancı ETF", "TR hisse fonu", "TR hisse", "Yabancı hisse"],
    "Tahvil": ["Diğer fon", "Borçlanma araçları fonu", "Eurobond"],
    "Nakit": ["Para piyasası fonu", "Döviz", "Türk Lirası"],
    "Emtia & Kripto": ["Kripto", "Altın", "Gümüş"]
}

def categorize_holdings_for_mekko(holdings, prices):
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

###############################################################################
# Here we fix the exchange_rates reference in create_mekko_chart
###############################################################################
def create_mekko_chart(investments):
    # ✅ Instead of referencing global "exchange_rates", fetch from session_state
    rates = st.session_state.get("exchange_rates", {})
    usdtry_rate = rates.get("USDTRY=X", 1)

    asset_class_totals = {asset: sum(sub.values()) for asset, sub in investments.items()}
    asset_class_totals = {k: v for k, v in asset_class_totals.items() if v > 0}

    total_investment_try = sum(asset_class_totals.values())
    total_investment_usd = total_investment_try / usdtry_rate

    if total_investment_try <= 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No investment data to plot.", ha="center", va="center")
        return fig, ax

    sorted_assets = sorted(asset_class_totals.items(), key=lambda x: x[1], reverse=True)
    sorted_investments = {
        asset: dict(sorted(investments[asset].items(), key=lambda x: x[1], reverse=True))
        for asset, _ in sorted_assets
    }

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

    plt.text(105, 120, f"Total: {total_investment_try:,.0f} TRY\n(~{total_investment_usd:,.0f} USD)",
             ha='right', va='top', fontsize=12, fontweight='bold', color='black')

    return fig, ax

###############################################################################
# 3. STREAMLIT APP LAYOUT
###############################################################################
def main():
    # Initialize the cookie manager using a secret key stored in Streamlit secrets.
    # Make sure you add a [COOKIE] section with COOKIE_SECRET_KEY in your Streamlit secrets file.

    cookies = EncryptedCookieManager(prefix="yatirimekko_",password = st.secrets["COOKIE"]["COOKIE_SECRET_KEY"])
    if not cookies.ready():
        st.stop()  # Wait until cookies are ready

    # If there's no user_id stored in the cookies, generate one and save it.
    if "user_id" not in cookies:
        cookies["user_id"] = str(uuid.uuid4())
        cookies.save()

    # Retrieve the persistent user id.
    persistent_user_id = cookies.get("user_id")

    if "retrieved_data" not in st.session_state:
        st.session_state["retrieved_data"] = retrieve_data(persistent_user_id)
        if st.session_state["retrieved_data"]:
            # 1. Preload the saved portfolio text
            st.session_state["portfolio_input"] = st.session_state["retrieved_data"].get("portfolio_input", "")
            
            # 2. Run the same calculation logic as the "Hesapla" button
            get_data()
            user_holdings = parse_user_holdings(st.session_state["portfolio_input"])
            classified_holdings = classify_assets(user_holdings, tefas_data)
            prices = fetch_prices(classified_holdings, tefas_data)
            net_worth = calculate_net_worth(classified_holdings, prices)

            portfolio_df = display_portfolio_table(classified_holdings, prices)
            investments = categorize_holdings_for_mekko(classified_holdings, prices)

            st.session_state["has_calculated"] = True
            st.session_state["portfolio_df"] = portfolio_df
            st.session_state["investments"] = investments
            st.session_state["net_worth"] = net_worth
            st.session_state["exchange_rates"] = exchange_rates


    # 1) Initialize session_state so results aren't lost if user modifies text
    if "has_calculated" not in st.session_state:
        st.session_state["has_calculated"] = False
    if "portfolio_df" not in st.session_state:
        st.session_state["portfolio_df"] = None
    if "investments" not in st.session_state:
        st.session_state["investments"] = None
    if "net_worth" not in st.session_state:
        st.session_state["net_worth"] = 0.0
    if "exchange_rates" not in st.session_state:
        st.session_state["exchange_rates"] = None

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
        
        Bitirdikten sonra **Hesapla** tuşuna bas ve portföy dağılımını incele.
        """,
        unsafe_allow_html=True
    )

    # SIDEBAR
    with st.sidebar:
        st.header("Portföyünü Gir")
        portfolio_text = st.text_area("Format: KOD ADET", height=200, key="portfolio_input")
        st.write(f"Anonim kullanıcı kimliğiniz: {persistent_user_id}")
        calc_button = st.button("Hesapla")

    # Only proceed with calculation if button is pressed
    if calc_button:
        if not portfolio_text.strip():
            st.warning("No portfolio data provided. Please enter something in the sidebar.")
        else:
            # CALL OUR NEW FUNCTION HERE, ONCE, AFTER USER PRESSES 'Hesapla'
            get_data()

            # PROCESS PORTFOLIO
            user_holdings = parse_user_holdings(portfolio_text)
            classified_holdings = classify_assets(user_holdings, tefas_data)
            prices = fetch_prices(classified_holdings, tefas_data)
            net_worth = calculate_net_worth(classified_holdings, prices)

            # Build the table and the investments dictionary
            portfolio_df = display_portfolio_table(classified_holdings, prices)
            investments = categorize_holdings_for_mekko(classified_holdings, prices)

            # Store results in session_state so they persist
            st.session_state["has_calculated"] = True
            st.session_state["portfolio_df"] = portfolio_df
            st.session_state["investments"] = investments
            st.session_state["net_worth"] = net_worth
            st.session_state["exchange_rates"] = exchange_rates

    # SHOW RESULTS IF ALREADY CALCULATED (OR JUST DONE)
    if st.session_state["has_calculated"]:
        st.subheader("Tablo Görünümü")
        st.data_editor(
            st.session_state["portfolio_df"],
            use_container_width=True,
            hide_index=True,
            column_config={"% Share": st.column_config.NumberColumn("% Share", format="%.1f%%")}
        )

        st.subheader("Mekko Görünümü")
        fig, ax = create_mekko_chart(st.session_state["investments"])
        st.pyplot(fig)

        # SIDEBAR: Upload after we have results
        with st.sidebar:
            st.markdown("---")
            upload_button = st.button("Sonuçları kaydet")
            if upload_button:
                if st.session_state["portfolio_df"] is None:
                    st.warning("Önce 'Hesapla' tuşuna basmalısınız!")
                else:
                    processed_json = st.session_state["portfolio_df"].to_json(orient="records")
                    data_to_insert = {
                        "user_id": persistent_user_id,
                        "portfolio_input": portfolio_text,
                        "processed_output": processed_json,
                        "net_worth": float(st.session_state["net_worth"]),
                        "usdtry_rate": float(st.session_state["exchange_rates"].get("USDTRY=X", 1))
                    }
                    try:
                        response = supabase.table("yatirimekko").insert(data_to_insert).execute()
                        st.success("Başarıyla kaydedildi!")
                    except Exception as e:
                        st.error(f"Hata oluştu: {e}")
    else:
        # If user never calculated, display helpful message
        st.info("Portföyünüzü girin ve 'Hesapla' tuşuna basın.")

    # Feedback Section
    st.markdown("---")
    st.subheader("Geri Bildirim")
    feedback_text = st.text_area("Uygulamayla ilgili önerilerinizi veya sorunlarınızı paylaşın:", key="feedback_input")

    if st.button("Gönder"):
        if not feedback_text.strip():
            st.warning("Lütfen bir geri bildirim girin!")
        else:
            feedback_data = {"user_id": persistent_user_id, "feedback": feedback_text}
            try:
                response = supabase.table("feedback").insert(feedback_data).execute()
                st.success("Geri bildiriminiz kaydedildi! Teşekkürler.")
            except Exception as e:
                st.error(f"Geri bildirim kaydedilirken hata oluştu: {e}")


if __name__ == "__main__":
    main()
