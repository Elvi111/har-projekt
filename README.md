# Forge: njohja e aktiviteteve nga sensorët e telefonit

Projekt për lëndën e Inteligjencës Artificiale (Grupi 24).
Modeli njeh gjashtë aktivitete (ecje, ngjitje shkallësh, zbritje shkallësh, ulur, në këmbë, shtrirë) dhe prototipi jep një këshillë kur dikush qëndron ulur shumë.

## Të dhënat
Përdorim dataset-in publik UCI "Human Activity Recognition Using Smartphones".
Shkarkojeni vetë dhe vendosni train.csv dhe test.csv në folderin `data/`.

## Si ta nisni
1. Instaloni bibliotekat: `pip install pandas scikit-learn matplotlib streamlit joblib notebook`
2. Hapni notebook-ët te folderi `notebooks/` dhe ekzekutoni qelizat nga lart poshtë.
3. Nisni aplikacionin: `python -m streamlit run app/app.py`

## Rezultatet
Random Forest 92.7%, KNN 90.4%, Decision Tree 86.2% (saktësia në pjesën e provës).