import streamlit as st
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="📁 CSV Editor", layout="wide")
st.title("📊 Editor CSV tabulek")

DATA_DIR = "data"

# Mapování názvů souborů na labely s ikonami
CSV_LABELS = {
    "businesses": "🏢 Podniky",
    "fields": "🟩 Pole",
    "pozemky": "🌍 Pozemky",
    "roky": "📆 Roky",
    "sbernamista": "📌 Sběrná místa",
    "sbernasrazky": "🌧️ Srážky",
    "sumplodiny": "📊 Souhrn plodin",
    "typpozemek": "📑 Typy pozemků",
    "userpodniky": "👥 Uživatelé a podniky",
    "users": "👤 Uživatelé",
    "crops": "🌾 Plodiny",
    "varieties_seed": "🌱 Odrůdy"
}

# Načíst dostupné soubory a filtrovat jen ty, které máme v mapě
available_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
available_keys = [os.path.splitext(f)[0] for f in available_files]
menu_items = [key for key in CSV_LABELS if key in available_keys]

# Statické menu
selected_key = st.sidebar.radio("📁 Vyber tabulku", menu_items, format_func=lambda k: CSV_LABELS[k])

# Načtení dat a editor
if selected_key:
    file_name = selected_key + ".csv"
    file_path = os.path.join(DATA_DIR, file_name)

    try:
        df = pd.read_csv(file_path)

        # Skrytí sloupce 'id' (pokud existuje)
        if 'id' in df.columns:
            id_col = df['id']
            df_visible = df.drop(columns=['id'])
        else:
            id_col = None
            df_visible = df

        st.subheader(f"{CSV_LABELS[selected_key]} (`{file_name}`)")

        # --- 🌾 Plodiny: tabulka + tlačítko pro editaci pořadí ---
     # --- 🌾 Plodiny: šipky + nové řádky (poradi & id) ---
        if selected_key == "crops":
            import numpy as np

            # Pracovní DF včetně 'id' (kvůli stabilnímu mapování)
            df_work = df.copy()

            # Schovat odruda (pokud je)
            if "odruda" in df_work.columns:
                df_work = df_work.drop(columns=["odruda"])

            # Y/N -> bool pro checkboxy
            for c in ["enable_main_table", "show_in_table"]:
                if c in df_work.columns:
                    df_work[c] = df_work[c].astype(str).str.upper().map({"Y": True, "N": False}).fillna(False)
                else:
                    df_work[c] = False

            # poradi jako číslo + doplnění chybějících
            if "poradi" not in df_work.columns:
                df_work["poradi"] = np.nan
            df_work["poradi"] = pd.to_numeric(df_work["poradi"], errors="coerce")
            start_ord = (int(df_work["poradi"].max()) + 1) if df_work["poradi"].notna().any() else 0
            for ix in df_work.index[df_work["poradi"].isna()]:
                df_work.at[ix, "poradi"] = start_ord
                start_ord += 1
            df_work["poradi"] = df_work["poradi"].astype(int)
            df_work = df_work.sort_values("poradi").reset_index(drop=True)

            # Stav do session (ponecháme i 'id', ale do editoru ho neukážeme)
            SKEY = "crops_state"
            if SKEY not in st.session_state:
                st.session_state[SKEY] = df_work.copy()

            # Panel pro přehazování (šipky mimo editor)
            def swap(i, j):
                dfw = st.session_state[SKEY]
                if 0 <= i < len(dfw) and 0 <= j < len(dfw):
                    a, b = int(dfw.loc[i, "poradi"]), int(dfw.loc[j, "poradi"])
                    dfw.loc[i, "poradi"], dfw.loc[j, "poradi"] = b, a
                    st.session_state[SKEY] = dfw.sort_values("poradi").reset_index(drop=True)

            with st.expander("↕️ Přesun pořadí", expanded=False):
                dfw = st.session_state[SKEY]
                for i, row in dfw.iterrows():
                    c1, c2, c3, c4 = st.columns([0.06, 0.06, 0.64, 0.24])
                    c1.button("⬆️", key=f"up_{i}", disabled=(i == 0), on_click=swap, args=(i, i-1))
                    c2.button("⬇️", key=f"down_{i}", disabled=(i == len(dfw)-1), on_click=swap, args=(i, i+1))
                    c3.write(row.get("nazev", f"řádek {i+1}"))
                    c4.write(f"pořadí: {int(row['poradi'])}")

            # Editor (bez 'id', 'poradi' jen na čtení – mění se šipkami; nové řádky se mohou přidat)
            df_for_editor = st.session_state[SKEY].drop(columns=["id"], errors="ignore").copy()
            edited_df = st.data_editor(
                df_for_editor,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "enable_main_table": st.column_config.CheckboxColumn("Ukázat v hlavním přehledu"),
                    "show_in_table": st.column_config.CheckboxColumn("Ukázat v tabulce"),
                    "poradi": st.column_config.NumberColumn("Pořadí", step=1, help="Mění se šipkami výše"),
                },
                disabled=["poradi"],  # pořadí se mění výše šipkami
            )

            # --- Uložení (ošetřené NOVÉ ŘÁDKY) ---
            if st.button("💾 Uložit změny"):
                state = st.session_state[SKEY].copy()     # má 'id' i aktuální 'poradi'
                out = edited_df.copy()

                # 1) Doplň poradi novým řádkům (NaN -> max+1..)
                out["poradi"] = pd.to_numeric(out.get("poradi", np.nan), errors="coerce")
                miss = out["poradi"].isna()
                if miss.any():
                    start = (int(state["poradi"].max()) + 1) if not state.empty else 0
                    seq = list(range(start, start + miss.sum()))
                    out.loc[miss, "poradi"] = seq
                out["poradi"] = out["poradi"].astype(int)

                # 2) Bool -> "Y"/"N"
                for c in ["enable_main_table", "show_in_table"]:
                    if c in out.columns:
                        out[c] = out[c].map(lambda x: "Y" if bool(x) else "N")

                # 3) Sloučení se stávajícím stavem a přiřazení ID novým řádkům
                #    (řídíme se 'poradi' – stav je již setříděný)
                #   a) aktualizuj existující řádky (podle pořadí)
                common_n = min(len(out), len(state))
                editable_cols = [c for c in out.columns if c in state.columns and c not in ["id", "poradi"]]
                if common_n:
                    state.loc[:common_n-1, editable_cols] = out.loc[:common_n-1, editable_cols].values
                    state.loc[:common_n-1, "poradi"] = out.loc[:common_n-1, "poradi"].values

                #   b) přidej nové řádky (ty, které přibyly v editoru)
                if len(out) > len(state):
                    new_rows = out.iloc[len(state):].copy()
                    # Vytvoř ID, pokud je máme v tabulce
                    if "id" in state.columns:
                        max_id = int(pd.to_numeric(state["id"], errors="coerce").max()) if not state.empty else 0
                        new_ids = list(range(max_id + 1, max_id + 1 + len(new_rows)))
                        new_rows.insert(0, "id", new_ids)
                    state = pd.concat([state, new_rows[state.columns]], ignore_index=True)

                # 4) finální seřazení a zápis
                state = state.sort_values("poradi").reset_index(drop=True)

                # pokud původní CSV mělo 'id', ponecháme; jinak sloupec nebude
                state.to_csv(file_path, index=False)
                st.success(f"Změny uloženy do `{file_name}` ✅")

            st.stop()



        # --- Generický editor pro ostatní tabulky ---
        edited_df = st.data_editor(df_visible, num_rows="dynamic", use_container_width=True)

        if st.button("💾 Uložit změny"):
            if id_col is not None:
                df_to_save = edited_df.copy()
                df_to_save.insert(0, 'id', id_col)
            else:
                df_to_save = edited_df
            df_to_save.to_csv(file_path, index=False)
            st.success(f"Změny uloženy do `{file_name}` ✅")

    except Exception as e:
        st.error(f"❌ Chyba při načítání `{file_name}`: {e}")