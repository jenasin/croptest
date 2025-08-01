import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📁 CSV Editor", layout="wide")
st.title("📊 Editor CSV tabulek")

DATA_DIR = "data"

# Mapování názvů souborů na labely s ikonami
CSV_LABELS = {
    "businesses": "🏢 Podniky",
    "crops": "🌾 Plodiny",
    "fields": "🟩 Pole",
    "pozemky": "🌍 Pozemky",
    "roky": "📆 Roky",
    "sbernamista": "📌 Sběrná místa",
    "sbernasrazky": "🌧️ Srážky",
    "sumplodiny": "📊 Souhrn plodin",
    "typpozemek": "📑 Typy pozemků",
    "userpodniky": "👥 Uživatelé a podniky",
    "users": "👤 Uživatelé",
    "varieties_seed": "🌱 Odrody / semena"
}

# Načíst dostupné soubory a filtrovat jen ty, které máme v mapě
available_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
available_keys = [os.path.splitext(f)[0] for f in available_files]
menu_items = [key for key in CSV_LABELS if key in available_keys]

# Statické menu
selected_key = st.sidebar.radio("📁 Vyber tabulku", menu_items, format_func=lambda k: CSV_LABELS[k])

# Načtení dat
if selected_key:
    file_name = selected_key + ".csv"
    file_path = os.path.join(DATA_DIR, file_name)

    try:
        df = pd.read_csv(file_path)
        st.subheader(f"{CSV_LABELS[selected_key]} (`{file_name}`)")
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Uložit změny"):
            edited_df.to_csv(file_path, index=False)
            st.success(f"Změny uloženy do `{file_name}`")

    except Exception as e:
        st.error(f"❌ Chyba při načítání `{file_name}`: {e}")
