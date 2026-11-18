import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

RRUGA = Path(__file__).resolve().parent.parent

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
}


@st.cache_resource
def ngarko_modelin():
    return joblib.load(RRUGA / "model.joblib")


@st.cache_data
def ngarko_te_dhenat():
    return pd.read_csv(RRUGA / "data" / "test.csv")


model = ngarko_modelin()
test = ngarko_te_dhenat()

st.title("Forge")
st.caption(
    "Prototip: njohja e aktiviteteve nga sensorët e telefonit. "
    "Është një simulim me matje nga dataset-i, jo telefoni i vërtetë."
)

skenari = st.selectbox("Zgjidh një skenar", list(SKENARET))
kufiri = st.slider("Pas sa matjesh radhazi 'ulur' japim këshillën?", 3, 10, 5)

if st.button("Nis simulimin"):
    pjeset = []
    for aktiviteti, numri in SKENARET[skenari]:
        mostrat = test[test["Activity"] == aktiviteti].sample(
            numri, random_state=len(pjeset)
        )
        pjeset.append(mostrat)
    të_dhënat = pd.concat(pjeset)

    X = të_dhënat.drop(columns=["subject", "Activity"])
    parashikimet = model.predict(X)

    radhazi = 0
    jo_ulur = 0
    keshilla_ne = None
    rreshtat = []
    for i, p in enumerate(parashikimet, start=1):
        if p == "SITTING":
            radhazi += 1
            jo_ulur = 0
        else:
            jo_ulur += 1
            if jo_ulur >= 2:
                radhazi = 0
        if radhazi >= kufiri and keshilla_ne is None:
            keshilla_ne = i
        rreshtat.append(
            {"Matja": i, "Aktiviteti i njohur": EMRAT[p], "Ulur radhazi": radhazi}
        )

    st.metric("Aktiviteti i fundit i njohur", EMRAT[parashikimet[-1]])

    if keshilla_ne:
        st.warning(
            f"Ke qëndruar ulur shumë (nga matja {keshilla_ne}). Ngrihu pak!"
        )
    else:
        st.success("Nuk u arrit kufiri i uljes. Vazhdo kështu!")

    st.subheader("Çfarë njohu modeli matje pas matjeje")
    st.dataframe(pd.DataFrame(rreshtat), hide_index=True)

    st.subheader("Përmbledhja")
    st.bar_chart(pd.Series([EMRAT[p] for p in parashikimet]).value_counts())