import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

RRUGA = Path(__file__).resolve().parent.parent
FEEDBACK = RRUGA / "feedback.csv"

EMRAT = {
    "LAYING": "Shtrirë",
    "SITTING": "Ulur",
    "STANDING": "Në këmbë",
    "WALKING": "Ecje",
    "WALKING_UPSTAIRS": "Ngjitje shkallësh",
    "WALKING_DOWNSTAIRS": "Zbritje shkallësh",
}

SKENARET = {
    "Student në mësim: ulur shumë": [
        ("SITTING", 8), ("STANDING", 2), ("WALKING", 3),
    ],
    "Ditë e zakonshme": [
        ("WALKING", 3), ("SITTING", 7), ("STANDING", 3),
        ("WALKING_UPSTAIRS", 2), ("SITTING", 4),
    ],
    "Pushim i mirë çdo orë": [
        ("SITTING", 4), ("WALKING", 3), ("SITTING", 4), ("WALKING", 3),
    ],
    "Mbrëmje e qetë në shtëpi": [
        ("STANDING", 3), ("SITTING", 5), ("LAYING", 6),
    ],
}

st.set_page_config(page_title="Forge", layout="centered")

st.markdown(
    """
    <style>
    .bllok {background:#FF6B4A; color:#1B1B1B; border-radius:0 0 36px 36px;
            padding:24px 24px 28px; margin-bottom:16px;}
    .bllok .fjala {font-size:88px; font-weight:900; line-height:1; letter-spacing:-3px;}
    .bllok .vogel {font-size:16px; font-weight:600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def ngarko_modelin():
    return joblib.load(RRUGA / "model.joblib")


@st.cache_data
def ngarko_te_dhenat():
    return pd.read_csv(RRUGA / "data" / "test.csv")


def ruaj_pergjigjen(parashikimi, e_vertete):
    rreshti = st.session_state["matjet"].iloc[[-1]].copy()
    rreshti.insert(0, "aktiviteti_i_vertete", e_vertete)
    rreshti.insert(0, "parashikimi", parashikimi)
    rreshti.insert(0, "koha", datetime.now().isoformat(timespec="seconds"))
    rreshti.to_csv(FEEDBACK, mode="a", header=not FEEDBACK.exists(), index=False)


model = ngarko_modelin()
test = ngarko_te_dhenat()

st.title("Forge")
st.caption(
    "Prototip: simulim me matje nga dataset-i UCI, jo me telefonin e vërtetë."
)

skenari = st.selectbox("Zgjidh një skenar", list(SKENARET))
kufiri = st.slider("Pas sa matjesh ulur radhazi japim këshillën?", 3, 10, 5)

if st.button("Nis simulimin"):
    pjeset = []
    for i, (aktiviteti, numri) in enumerate(SKENARET[skenari]):
        pjeset.append(test[test["Activity"] == aktiviteti].sample(numri, random_state=i))
    matjet = pd.concat(pjeset).drop(columns=["subject", "Activity"])
    st.session_state["matjet"] = matjet
    st.session_state["parashikimet"] = list(model.predict(matjet))
    st.session_state["probat"] = list(model.predict_proba(matjet).max(axis=1))
    st.session_state["feedback_hapur"] = False

if "parashikimet" in st.session_state:
    parashikimet = st.session_state["parashikimet"]
    probat = st.session_state["probat"]

    # Numërimi i kohës së ulur: zerohet vetëm pas 2 matjesh radhazi jo "ulur"
    radhazi, jo_ulur, keshilla_ne = 0, 0, None
    rreshtat = []
    for i, p in enumerate(parashikimet, start=1):
        if p == "SITTING":
            radhazi += 1
            jo_ulur = 0
        else:
            jo_ulur += 1
            if jo_ulur >= 2:
                radhazi = 0
        keshille = radhazi >= kufiri
        if keshille and keshilla_ne is None:
            keshilla_ne = i
        rreshtat.append({
            "Matja": i,
            "Aktiviteti": EMRAT[p],
            "Ulur radhazi": radhazi,
            "Këshillë": "Po" if keshille else "",
        })

    e_fundit = parashikimet[-1]
    st.markdown(
        f"""<div class="bllok"><div class="vogel">Tani po bën</div>
        <div class="fjala">{EMRAT[e_fundit]}</div>
        <div class="vogel">Jemi {round(probat[-1] * 100)}% të sigurt.</div></div>""",
        unsafe_allow_html=True,
    )

    st.write(f"**Ulur pa u ngritur:** {radhazi} nga {kufiri} matje")
    st.progress(min(radhazi / kufiri, 1.0))
    if radhazi >= kufiri:
        st.warning("Ke qëndruar ulur shumë. Ngrihu pak!")
    elif keshilla_ne:
        st.info(f"Këshilla u dha në matjen {keshilla_ne}. Pastaj u ngrite, shumë mirë!")
    else:
        st.success("Nuk u arrit kufiri i uljes. Vazhdo kështu!")

    st.subheader(f'A po bën vërtet "{EMRAT[e_fundit]}"?')
    k1, k2 = st.columns(2)
    if k1.button("Po, saktë"):
        ruaj_pergjigjen(e_fundit, e_fundit)
        st.success("Faleminderit!")
    if k2.button("Jo, po bëj tjetër"):
        st.session_state["feedback_hapur"] = True

    if st.session_state.get("feedback_hapur"):
        zgjedhja = st.selectbox(
            "Çfarë po bën vërtet?", list(EMRAT), format_func=lambda k: EMRAT[k]
        )
        if st.button("Ruaj"):
            ruaj_pergjigjen(e_fundit, zgjedhja)
            st.session_state["feedback_hapur"] = False
            st.success("Faleminderit! Përgjigjja jote u ruajt.")

    if FEEDBACK.exists():
        st.caption(f"Përgjigje të ruajtura deri tani: {len(pd.read_csv(FEEDBACK))}")

    st.subheader("Përmbledhja")
    st.bar_chart(pd.Series([EMRAT[p] for p in parashikimet]).value_counts())
    with st.expander("Shiko matje pas matjeje"):
        st.dataframe(pd.DataFrame(rreshtat), hide_index=True)