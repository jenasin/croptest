import streamlit as st
import pandas as pd
import os
import numpy as np
import os, hashlib, binascii
import hmac


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


def make_password_hash(password: str, *, iterations: int = 200_000) -> tuple[str, str, int]:
    """PBKDF2-HMAC-SHA256: vrátí (salt_hex, hash_hex, iterations)."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return binascii.hexlify(salt).decode(), binascii.hexlify(dk).decode(), iterations

def verify_password(password: str, salt_hex: str, hash_hex: str, iterations: int) -> bool:
    """Ověření hesla proti uloženému salt+hash."""
    salt = binascii.unhexlify(salt_hex.encode())
    expected_hash = binascii.unhexlify(hash_hex.encode())
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected_hash)

def csv_ids_to_list(s):
    if s is None or (isinstance(s, float) and pd.isna(s)) or str(s).strip()=="":
        return []
    return [int(x) for x in str(s).replace(" ", "").split(",") if str(x).strip().isdigit()]

def list_to_csv_ids(lst):
    if not lst:
        return ""
    return ",".join(str(int(x)) for x in lst)

# ===== USERS: kompletní a robustní správa (jen 3 role: admin/watcher/user) =====
if selected_key == "users":
    # --- Načti businesses kvůli výběru podniků (bez pádů i když chybí) ---
    biz_path = os.path.join(DATA_DIR, "businesses.csv")
    biz_df = None
    if os.path.exists(biz_path):
        try:
            biz_df = pd.read_csv(biz_path)
        except Exception:
            biz_df = None

    biz_choices, biz_id_to_name, biz_name_to_id = [], {}, {}
    if biz_df is not None and not biz_df.empty:
        # detekce sloupců id / název (fallback na 1./2. sloupec)
        cand_id = [c for c in biz_df.columns if c.lower() in ("id", "business_id")]
        cand_name = [c for c in biz_df.columns if c.lower() in ("name", "nazev", "title")]
        biz_id_col = cand_id[0] if cand_id else biz_df.columns[0]
        biz_name_col = cand_name[0] if cand_name else (biz_df.columns[1] if len(biz_df.columns) > 1 else biz_df.columns[0])
        biz_df[biz_id_col] = pd.to_numeric(biz_df[biz_id_col], errors="coerce").astype("Int64")
        for _, r in biz_df.iterrows():
            if pd.isna(r[biz_id_col]):
                continue
            bid, bname = int(r[biz_id_col]), str(r[biz_name_col])
            biz_choices.append((bid, bname))
        biz_id_to_name = {i: n for i, n in biz_choices}
        biz_name_to_id = {n: i for i, n in biz_choices}

    # --- Users DF z právě načteného df (z tvého hlavního kódu) ---
    users_df = df.copy()

    # Jistota: vytvoř chybějící sloupce, ať to nikdy nepadá
    for need_col, default in [
        ("username", ""),
        ("full_name", ""),
        ("email", ""),
        ("role", "user"),
        ("business_ids", ""),        # <<< zajistí, že existuje
        ("password_salt", ""),
        ("password_hash", ""),
        ("password_iters", ""),      # ukládáme i iterace PBKDF2
        ("is_active", True),
    ]:
        if need_col not in users_df.columns:
            users_df[need_col] = default

    # id série (pokud existuje)
    id_series = users_df["id"] if "id" in users_df.columns else None

    # Tabulka pro zobrazení (id skryjeme)
    show_df = users_df.drop(columns=["id"], errors="ignore").copy()

    # Čitelné názvy podniků (jen pro zobrazení)
    if biz_id_to_name:
        show_df["Podniky (názvy)"] = show_df["business_ids"].apply(
            lambda s: ", ".join(biz_id_to_name.get(i, f"#{i}") for i in csv_ids_to_list(s))
        )

    # --- Povolené role: přesně 3 ---
    allowed_roles = ["admin", "watcher", "user"]
    show_df["role"] = show_df["role"].astype(str)
    show_df.loc[~show_df["role"].isin(allowed_roles), "role"] = "user"
    roles = allowed_roles

    # --- Editor pro běžná pole (bez hesel) ---
    edited_tbl = st.data_editor(
        show_df,
        num_rows="fixed",         # přidávání níže přes formulář
        use_container_width=True,
        hide_index=True,
        column_config={
            "username": st.column_config.TextColumn("Uživatelské jméno"),
            "full_name": st.column_config.TextColumn("Jméno a příjmení"),
            "email": st.column_config.TextColumn("E-mail"),
            "role": st.column_config.SelectboxColumn("Role", options=roles),
            "business_ids": st.column_config.TextColumn("Podniky (ID, čárkami)"),
            "Podniky (názvy)": st.column_config.TextColumn("Podniky (názvy)"),
            "is_active": st.column_config.CheckboxColumn("Aktivní"),
            "password_salt": st.column_config.TextColumn("password_salt"),
            "password_hash": st.column_config.TextColumn("password_hash"),
            "password_iters": st.column_config.TextColumn("password_iters"),
        },
        disabled=["Podniky (názvy)", "password_salt", "password_hash", "password_iters"],
    )

    if st.button("💾 Uložit změny (bez hesel)"):
        out = edited_tbl.copy()
        # Drop pouze zobrazovacího sloupce, ať se nedostane do CSV
        if "Podniky (názvy)" in out.columns:
            out = out.drop(columns=["Podniky (názvy)"])
        # Zarovnej role do allowed setu
        if "role" in out.columns:
            out["role"] = out["role"].astype(str)
            out.loc[~out["role"].isin(allowed_roles), "role"] = "user"
        # Normalizuj business_ids
        if "business_ids" not in out.columns:
            out["business_ids"] = ""
        out["business_ids"] = out["business_ids"].apply(csv_ids_to_list).apply(list_to_csv_ids)
        # Jistota heslových sloupců
        for need_col in ["password_salt", "password_hash", "password_iters"]:
            if need_col not in out.columns:
                out[need_col] = ""
        # Vrať id dopředu (pokud máme)
        if id_series is not None:
            out.insert(0, "id", id_series)
        out.to_csv(file_path, index=False)
        st.success(f"Změny uloženy do `{file_name}` ✅")
        st.experimental_rerun()

    st.markdown("---")

    # --- Přidat nového uživatele (hash hesla PBKDF2) ---
    with st.expander("➕ Přidat uživatele", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            new_username = st.text_input("Uživatelské jméno*", key="new_u")
            new_fullname = st.text_input("Jméno a příjmení", key="new_f")
            new_email = st.text_input("E-mail", key="new_e")
        with c2:
            new_role = st.selectbox("Role", options=roles, index=roles.index("user"), key="new_r")
            # výběr podniků (preferuj názvy, jinak ruční ID)
            if biz_choices:
                sel_biz_names = st.multiselect("Podniky", options=[n for _, n in biz_choices], key="new_b")
                sel_biz_ids = [biz_name_to_id[n] for n in sel_biz_names]
            else:
                sel_biz_ids = csv_ids_to_list(st.text_input("Podniky (ID, čárkami)", key="new_b_ids"))

        pw1 = st.text_input("Heslo*", type="password", key="new_p1")
        pw2 = st.text_input("Potvrzení hesla*", type="password", key="new_p2")
        active_flag = st.checkbox("Aktivní", value=True, key="new_active")

        if st.button("📥 Uložit nového uživatele"):
            if not new_username or not pw1 or not pw2:
                st.error("Vyplň *Uživatelské jméno* a obě pole *Heslo*.")
            elif pw1 != pw2:
                st.error("Hesla se neshodují.")
            else:
                curr = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()

                # Jistota: existují očekávané sloupce
                for col, default in [
                    ("username", ""), ("full_name", ""), ("email", ""), ("role", "user"),
                    ("business_ids", ""), ("password_salt", ""), ("password_hash", ""), ("password_iters", ""), ("is_active", True),
                ]:
                    if col not in curr.columns:
                        curr[col] = default

                # Unikátní username
                if "username" in curr.columns and new_username in curr["username"].astype(str).values:
                    st.error("Uživatel s tímto uživatelským jménem už existuje.")
                else:
                    # Hash hesla (tvá funkce vrací 3 hodnoty)
                    salt_hex, hash_hex, iters = make_password_hash(pw1)

                    # Nové ID (pokud má CSV sloupec id)
                    new_id = None
                    if "id" in curr.columns:
                        max_id = pd.to_numeric(curr["id"], errors="coerce").max()
                        new_id = (int(max_id) + 1) if pd.notna(max_id) else 1

                    # Slož nový záznam
                    new_user = {
                        "username": new_username,
                        "full_name": new_fullname,
                        "email": new_email,
                        "role": new_role if new_role in allowed_roles else "user",
                        "business_ids": list_to_csv_ids(sel_biz_ids),
                        "password_salt": salt_hex,
                        "password_hash": hash_hex,
                        "password_iters": iters,
                        "is_active": bool(active_flag),
                    }
                    if new_id is not None:
                        new_user["id"] = new_id

                    # Dorovnej chybějící sloupce dle CSV
                    for col in curr.columns:
                        new_user.setdefault(col, "" if col != "is_active" else True)

                    # Pokud má new_user sloupec navíc, přidej ho do curr
                    extra_cols = [c for c in new_user.keys() if c not in curr.columns]
                    if extra_cols:
                        curr = curr.reindex(columns=list(curr.columns) + extra_cols)

                    # Přidej řádek a ulož
                    curr = pd.concat([curr, pd.DataFrame([new_user])[curr.columns]], ignore_index=True)
                    curr.to_csv(file_path, index=False)
                    st.success("Nový uživatel uložen ✅")
                    st.experimental_rerun()

    # --- Změna hesla existujícího uživatele ---
    with st.expander("🗝️ Změnit heslo existujícího uživatele", expanded=False):
        user_choices = users_df["username"].astype(str).tolist() if "username" in users_df.columns else []
        sel_user = st.selectbox("Uživatel", options=user_choices)
        npw1 = st.text_input("Nové heslo*", type="password", key="pw1")
        npw2 = st.text_input("Potvrzení nového hesla*", type="password", key="pw2")
        if st.button("🔐 Uložit nové heslo"):
            if not sel_user or not npw1 or not npw2:
                st.error("Vyplň uživatele a obě pole hesla.")
            elif npw1 != npw2:
                st.error("Hesla se neshodují.")
            else:
                curr = pd.read_csv(file_path)
                if "username" not in curr.columns:
                    st.error("Soubor nemá sloupec 'username'.")
                else:
                    idx = curr.index[curr["username"].astype(str) == sel_user]
                    if len(idx) == 0:
                        st.error("Uživatel nenalezen.")
                    else:
                        salt_hex, hash_hex, iters = make_password_hash(npw1)
                        for col in ("password_salt", "password_hash", "password_iters"):
                            if col not in curr.columns:
                                curr[col] = ""
                        curr.loc[idx, "password_salt"] = salt_hex
                        curr.loc[idx, "password_hash"] = hash_hex
                        curr.loc[idx, "password_iters"] = iters
                        curr.to_csv(file_path, index=False)
                        st.success("Heslo změněno ✅")
                        st.experimental_rerun()

    # --- Změna podniků u existujícího uživatele (multiselect) ---
    with st.expander("🏢 Upravit podniky u uživatele", expanded=False):
        if not biz_choices:
            st.info("Soubor `businesses.csv` nebyl nalezen nebo je prázdný – uprav `business_ids` přímo v tabulce.")
        else:
            user_choices2 = users_df["username"].astype(str).tolist() if "username" in users_df.columns else []
            sel_user2 = st.selectbox("Uživatel", options=user_choices2, key="biz_user_sel")

            curr2 = pd.read_csv(file_path)
            # Jistota existence sloupce business_ids
            if "business_ids" not in curr2.columns:
                curr2["business_ids"] = ""

            row = curr2[curr2["username"].astype(str) == sel_user2]
            if row.empty:
                st.warning("Uživatel nenalezen.")
            else:
                row = row.iloc[0]
                current_ids = csv_ids_to_list(row["business_ids"])
                current_names = [biz_id_to_name.get(i, f"#{i}") for i in current_ids]
                chosen_names = st.multiselect("Podniky", options=[n for _, n in biz_choices], default=current_names)
                if st.button("💼 Uložit podniky"):
                    chosen_ids = [biz_name_to_id[n] for n in chosen_names]
                    curr2.loc[curr2["username"].astype(str) == sel_user2, "business_ids"] = list_to_csv_ids(chosen_ids)
                    curr2.to_csv(file_path, index=False)
                    st.success("Podniky uloženy ✅")
                    st.experimental_rerun()

    st.stop()
# ===== /USERS =====

