import streamlit as st
st.title("Mon premier Streamlit")
st.write("Introduction")
if st.checkbox("Afficher"):
  st.write("Suite du Streamlit")
  
# Chargement des librairies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import pypickle


from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler

# Chargement du jeu de données

df = pd.read_csv("trafic_cycliste.csv", encoding="utf-8")


# Affichage des cinq premières lignes du dataframe
# df.head()

# Analyses préliminaires

# Affichage des principales informations du df, ce qui comprend notamment la typologie des variables
# df.info()

# Taux de NA de chaque colonne
# df.isna().mean() * 100


# Nombre de compteurs différents
print('Nombre de compteurs :', df['Nom du compteur'].nunique())

# Nombre de sites de comptages (d'adresses)
print('Nombre de sites de comptages : ', df['Nom du site de comptage'].nunique())

# Ces éléments sont importants car ils représentent la clé de voûte de notre analyse.

# Nombre d'heures total et nombre moyen de mesures par site
df_grpby = df.groupby(['Nom du site de comptage']).agg({'Comptage horaire':['mean', 'sum']})
print("Comptage horaire par site de comptage, dans l'ordre décroissant : \n", df_grpby.sort_values(by = ("Comptage horaire", "mean"), ascending = False), "\n")


# Analyse de la colonne Comptage horaire (seule colonne de données continues, potentielle variable cible du ML) avec la fonction "describe" qui permet de faire état de statistiques descriptives de base
df['Comptage horaire'].describe()

# Préparation du jeu de données & enrichissement


# Suppression des colonnes inutiles à l'analyse

df = df.drop(["Lien vers photo du site de comptage","test_lien_vers_photos_du_site_de_comptage_","id_photo_1","url_sites","type_dimage","ID Photos","Identifiant du compteur","Identifiant du site de comptage","Identifiant technique compteur"],axis=1)
# df.head()

# Analyse des valeurs manquantes sur nos colonnes
# df.isna().sum()


# Extraction des lignes contenant des NaN dans un df à part
df_na = df[df['Nom du site de comptage'].isna()]
compteurs_na = df_na['Nom du compteur'].unique()

# print("Nom des compteurs avec des valeurs NaN :", compteurs_na)


# Normalisation des valeurs
df['Coordonnées géographiques'] = df['Coordonnées géographiques'].str.replace(r'\s*,\s*', ',', regex=True)

# Identification des valeurs de remplacement
dict_sites = {
    "Face au 48 quai de la marne": {
        "coordonnees": df[df['Nom du site de comptage'] == 'Face au 48 quai de la marne']['Coordonnées géographiques'].unique()[0],
        "date": df[df['Nom du site de comptage'] == 'Face au 48 quai de la marne']["Date d'installation du site de comptage"].unique()[0]
    },
    "Pont des Invalides": {
        "coordonnees": df[df['Nom du compteur'] == 'Pont des Invalides N-S']['Coordonnées géographiques'].unique()[0],
        "date": df[df['Nom du compteur'] == 'Pont des Invalides N-S']["Date d'installation du site de comptage"].unique()[0]
    },
    "27 quai de la Tournelle": {
        "coordonnees": df[df['Nom du site de comptage'] == '27 quai de la Tournelle']['Coordonnées géographiques'].unique()[0],
        "date": df[df['Nom du site de comptage'] == '27 quai de la Tournelle']["Date d'installation du site de comptage"].unique()[0]
    },
    "Quai des Tuileries": {
        "coordonnees": df[df['Nom du site de comptage'] == 'Quai des Tuileries']['Coordonnées géographiques'].unique()[0],
        "date": df[df['Nom du site de comptage'] == 'Quai des Tuileries']["Date d'installation du site de comptage"].unique()[0]
    }
}

# Mapping compteurs - sites
dict_compteurs_sites = {
    "Face au 48 quai de la marne": ['Face au 48 quai de la marne NE-SO', 'Face au 48 quai de la marne SO-NE'],
    "Pont des Invalides": ['Pont des Invalides N-S'],
    "27 quai de la Tournelle": ['27 quai de la Tournelle NO-SE', '27 quai de la Tournelle SE-NO'],
    "Quai des Tuileries": ['Quai des Tuileries NO-SE', 'Quai des Tuileries SE-NO']
}

# Complétion des valeurs manquantes
for site, compteurs in dict_compteurs_sites.items():
    coordonnees = dict_sites[site]['coordonnees']
    date_installation = dict_sites[site]['date']

    condition = df['Nom du compteur'].isin(compteurs)
    df.loc[condition, 'Nom du site de comptage'] = site
    df.loc[condition, 'Coordonnées géographiques'] = coordonnees
    df.loc[condition, "Date d'installation du site de comptage"] = date_installation

# Vérification du nombre de valeurs NaN
# df.isna().sum()


# Conversion au format datetime
df["Date et heure de comptage"] = pd.to_datetime(df["Date et heure de comptage"], utc=True)
df["Date d'installation du site de comptage"] = pd.to_datetime(df["Date d'installation du site de comptage"])
df["mois_annee_comptage"] = pd.to_datetime(df["mois_annee_comptage"])


# Ajout de colonnes temporelles
df["Heure"] = df["Date et heure de comptage"].dt.strftime('%H' + "h")
df["Mois"] = df["mois_annee_comptage"].dt.month_name()
df['Jour'] = df['Date et heure de comptage'].dt.day_name()

# On affiche les cinq premières lignes du df avec les nouvelles colonnes
# df.head()


# Ajout d'une colonne correspondant aux périodes de vacances
vacances = []

periodes_vacances = [
    ("Grandes vacances 2023", date(2023, 7, 8), date(2023, 9, 3)),
    ("Toussaint 2023", date(2023, 10, 21), date(2023, 11, 5)),
    ("Noël 2023", date(2023, 12, 23), date(2024, 1, 7)),
    ("Hiver 2024", date(2024, 2, 10), date(2024, 2, 25)),
    ("Printemps 2024", date(2024, 4, 6), date(2024, 4, 21)),
    ("Grandes vacances 2024", date(2024, 7, 6), date(2024, 9, 1)),
    ("Toussaint 2024", date(2024, 10, 19), date(2024, 11, 3)),
    ("Noël 2024", date(2024, 12, 21), date(2025, 1, 5))
]

for nom_periode, start_date, end_date in periodes_vacances:
    current_date = start_date
    while current_date <= end_date:
        vacances.append(current_date)
        current_date += timedelta(days=1)

df['vacances_zone_C'] = df['Date et heure de comptage'].apply(lambda x: 1 if x.date() in set(vacances) else 0)

# Ajout d'une colonne correspondant aux jours fériés

jours_feries=[
    date(2024,1,1), # 1er Janvier 2024
    date(2024,4,1), # Lundi de Pâques 2024
    date(2024,5,1), # 1er mai 2024
    date(2024,5,8), # 8 mai 2024
    date(2024,5,9), # Ascension 2024
    date(2024,5,20), # Lundi de pentecôte
    date(2024,7,14), # Le 14 juillet 2024
    date(2023,8,15),date(2024,8,15), # Assomption 2023 et 2024
    date(2023,11,1),date(2024,11,1), # Toussaint 2023 et 2024
    date(2023,11,11),date(2024,11,11), # 11 novembre 2023 et 2024
    date(2023,12,25),date(2024,12,25) # Jour de Noêl 2023 et 2024
]

df['jours_feries']=df['Date et heure de comptage'].apply(lambda x: 1 if x.date() in jours_feries else 0)

# Data Visualisation

## Analyses temporelles

# Analyse du trafic global sur une année, jour par jour

fig, ax = plt.subplots(figsize=(15, 6))
df["jour_mois_annee_comptage"] = df["Date et heure de comptage"].dt.date
trafic_moyen_par_jour = df.groupby('jour_mois_annee_comptage').agg({"Comptage horaire" : "mean"}).reset_index()
sns.lineplot(data = trafic_moyen_par_jour, x = "jour_mois_annee_comptage", y = "Comptage horaire", color="royalblue")

jours_feries_dates = df[df['jours_feries'] == 1]["jour_mois_annee_comptage"].unique()

for jour_ferie in jours_feries_dates:
    ax.axvline(jour_ferie, color="red", linestyle="--", linewidth=2, alpha=0.3, label="Jour férié" if jour_ferie == jours_feries_dates[0] else "")

vacances_dates = df[df['vacances_zone_C'] == 1]["jour_mois_annee_comptage"].unique()

for start_date in vacances_dates:
    ax.axvspan(start_date, start_date, color="yellow", alpha=0.3, label="Vacances Zone C" if start_date == vacances_dates[0] else "")

df = df.drop(["jour_mois_annee_comptage"],axis=1)

plt.title("Comptage horaire moyen chaque jour \n", fontsize = 14)
plt.xlabel("Date", fontsize = 10)
plt.ylabel("Comptage horaire moyen", fontsize = 10)
plt.legend(loc = "best");


# Visualisation des passages horaires, chaque jour de la semaine, chaque heure de la journée

ordre_jour = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heatmap_jour_heure = df.groupby(['Jour', 'Heure']).agg({'Comptage horaire': 'mean'}).reset_index()
heatmap_jour_heure['Jour'] = pd.Categorical(heatmap_jour_heure['Jour'], categories=ordre_jour, ordered=True)
heatmap_jour_heure = heatmap_jour_heure.pivot(index="Jour", columns="Heure", values="Comptage horaire")

plt.figure(figsize=(18, 6))
sns.heatmap(heatmap_jour_heure, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.5)
plt.title("Heatmap des passages horaires par jour et par heure \n");

# Analyse du trafic mensuel : Comptage horaire moyen au fil des mois

fig = plt.figure(figsize = (8, 5))
trafic_moyen_par_mois = df.groupby('Mois').agg({"Comptage horaire" : "mean"}).reset_index()
ordre_mois = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
trafic_moyen_par_mois['Mois'] = pd.Categorical(trafic_moyen_par_mois['Mois'], categories=ordre_mois, ordered=True)

ax = sns.barplot(data = trafic_moyen_par_mois, x = "Mois", y = "Comptage horaire", hue = "Mois")
for p in ax.patches:
  ax.annotate(f'{p.get_height():.2f}',
              (p.get_x() + p.get_width() / 2, p.get_height()),
              ha='center',
              fontsize=8,
              xytext=(0, 5),
              textcoords='offset points')
plt.title("Comptage horaire moyen chaque mois \n", fontsize = 16)
plt.xlabel("Mois", fontsize = 12)
plt.ylabel("Nombre de passages moyen", fontsize = 12)
plt.xticks(rotation = 45);


# Comparaison entre Jour de repos VS Jours normaux, chaque mois
df['type_de_jour'] = df.apply(lambda row: 'Jour de repos' if (row['Jour'] == 'Sunday' or row['vacances_zone_C'] == 1 or row['jours_feries'] == 1) else 'Jour de travail', axis=1)

df_vac = df.groupby(['Mois', 'type_de_jour']).agg({'Comptage horaire': 'mean'}).reset_index()
df_vac['Mois'] = pd.Categorical(df_vac['Mois'], categories=ordre_mois, ordered=True)

fig = plt.figure(figsize = (10, 5))
sns.barplot(data= df_vac, x='Mois', y='Comptage horaire', hue='type_de_jour')

plt.title("Comparaison du comptage horaire entre Vacances/Fériés et Jours Normaux (par Mois)", fontsize=14)
plt.xlabel("Mois", fontsize=12)
plt.ylabel("Comptage horaire moyen", fontsize=12)
plt.xticks(rotation = 45)
plt.legend(loc = "best");


# Compteurs avec le plus et le moins de trafic

trafic_comptage = df.groupby(['Nom du compteur']).agg({'Comptage horaire': 'mean'}).sort_values(by = "Comptage horaire", ascending = False)
top_compteurs = trafic_comptage.head()
bottom_compteurs = trafic_comptage.tail()

fig, axes = plt.subplots(2, 1, figsize = (12, 10), sharex = True)

# Top 5
ax0 = sns.barplot(data = top_compteurs.reset_index(), x = "Comptage horaire", y = "Nom du compteur", ax = axes[0], hue = "Nom du compteur", palette = "Greens")
axes[0].set_title("Top 5 compteurs avec le plus de passages horaires \n")
axes[0].set_xlabel("Comptage horaire moyen \n")
for p in ax0.patches:
  ax0.annotate(f'{p.get_width():.2f}',
              (p.get_width(), p.get_y() + p.get_height() / 2),
              ha='left',
              va ='center',
              fontsize=8)

# Bottom 5
ax1 = sns.barplot(data = bottom_compteurs.reset_index(), x = "Comptage horaire", y = "Nom du compteur", ax = axes[1], hue = "Nom du compteur", palette = "Reds")
axes[1].set_title("\n Top 5 compteurs avec le moins de passages horaires \n");
for p in ax1.patches:
  ax1.annotate(f'{p.get_width():.2f}',
              (p.get_width(), p.get_y() + p.get_height() / 2),
              ha='left',
              va ='center',
              fontsize=8)


# Visualisation de l'évolution du nombre de sites de comptages installés depuis 2012

fig = plt.figure(figsize = (8, 5))
installation_comptage = df.groupby("Date d'installation du site de comptage")["Nom du site de comptage"].nunique().reset_index()
installation_comptage['Nombre de site de comptage cumulé'] = installation_comptage['Nom du site de comptage'].cumsum()
sns.lineplot(data = installation_comptage, x = "Date d'installation du site de comptage", y = "Nombre de site de comptage cumulé", drawstyle = "steps-post", color = "teal")

plt.title("Evolution de l'installation des sites de comptage \n", fontsize = 14)
plt.xlabel("Date d'installation", fontsize = 10);

## Analyses cartographiques

st.title("Analyses géographiques - Cartes")

st.write("Passons maintenant à une analyse géographique et observons les zones importantes d'affluence.")

st.header("Trafic cycliste moyen sur l'ensemble des jours")
# Comptage horaire moyen, sans condition de date.
trafic_comptage = df.groupby(['Nom du site de comptage', 'Coordonnées géographiques'])['Comptage horaire'].mean().reset_index()

top_sites = trafic_comptage.nlargest(5, 'Comptage horaire')
bottom_sites = trafic_comptage.nsmallest(5, 'Comptage horaire')

paris = folium.Map(location=[48.855578, 2.331830], zoom_start=12)

for index, site in trafic_comptage.iterrows():
    latitude, longitude = map(float, site['Coordonnées géographiques'].split(','))
    taille_cercle = site['Comptage horaire'] / 10

    if site['Nom du site de comptage'] in top_sites['Nom du site de comptage'].tolist():
        couleur = 'green'
    elif site['Nom du site de comptage'] in bottom_sites['Nom du site de comptage'].tolist():
        couleur = 'red'
    else:
        couleur = 'blue'

    folium.CircleMarker(
        [latitude, longitude],
        radius=taille_cercle,
        popup=f"{site['Nom du site de comptage']} - {round(site['Comptage horaire'], 2)} vélos/heure",
        tooltip=site['Nom du site de comptage'],
        color=couleur,
        fill=True,
        fill_opacity=0.3
    ).add_to(paris)

folium_static(paris, width=700, height=500)

st.write("Analyse :")
st.write("Sur la carte ci-dessus, la taille des cercles nous donne des informations sur l'affluence sur chacun des sites. Nous avons d'ailleurs choisi de mettre les sites de comptages avec le plus d'affluence en vert et à l'inverse, ceux avec le moins d'affluence en rouge, pour les mettre en relief.")

st.markdown("""
- Zones connaissant le plus d’affluence :
  - Les cinq sites de comptage enregistrant le plus de passages se trouvent plutôt au centre de Paris et côté rive droite où se trouvent de nombreux quartiers animés.
- Zones connaissant le moins d’affluence :
  - Les cinq sites enregistrant le moins de passages se trouvent en périphérie, au niveau des portes de la ville, où l'on trouve davantage de quartiers calmes et résidentiels.
""")

st.write("Essayons d'analyser cela plus en profondeur, en filtrant sur des heures et des jours spécifiques. Commençons par retirer les vacances, les jours fériés et les dimanches pour avoir les jours de travail et en se focalisant sur l'état à 7h, le matin.")

# Carte de Paris à 7h, les jours de travail
st.header("Trafic cycliste à 7h en semaine (hors vacances et jours fériés)")


df_7h = df[(df['Heure'] == "07h") & (df["type_de_jour"] == "Jour de travail")]

trafic_comptage_7h = df_7h.groupby(['Nom du site de comptage', 'Coordonnées géographiques'])['Comptage horaire'].mean().reset_index()

top_sites = trafic_comptage_7h.nlargest(5, 'Comptage horaire')
bottom_sites = trafic_comptage_7h.nsmallest(5, 'Comptage horaire')

paris = folium.Map(location=[48.855578, 2.331830], zoom_start=12)

for index, site in trafic_comptage_7h.iterrows():
    latitude, longitude = map(float, site['Coordonnées géographiques'].split(','))
    taille_cercle = site['Comptage horaire'] / 10

    if site['Nom du site de comptage'] in top_sites['Nom du site de comptage'].tolist():
        couleur = 'green'
    elif site['Nom du site de comptage'] in bottom_sites['Nom du site de comptage'].tolist():
        couleur = 'red'
    else:
        couleur = 'blue'

    folium.CircleMarker(
        [latitude, longitude],
        radius=taille_cercle,
        popup=f"{site['Nom du site de comptage']} - {round(site['Comptage horaire'], 2)} vélos/heure",
        tooltip=site['Nom du site de comptage'],
        color=couleur,
        fill=True,
        fill_opacity=0.3
    ).add_to(paris)


folium_static(paris, width=700, height=500)

st.write("Analyse :")
st.write("En comparant cette carte avec la première, nous remarquons directement qu'en moyenne, les cercles sont bien plus grands (nous avons évidemment appliqué la même échelle qu'avant).") 
st.write("Les zones au centre, côté rive droite, qui connaissaient déjà des pics d’affluence sur l’ensemble des jours connaissent davantage d’affluence.")
st.write("D’autres zones émergent en revanche. Nous observons que, à 7h du matin en semaine, nos anciens faibles sites de comptage (en périphérie) ont désormais une affluence au moins moyenne. Nous pouvons faire la supposition que cela est dû aux parisiens excentrés venant sur leur lieu de travail.")

st.write("Examinons maintenant l'état de Paris, les jours de repos à 14h, où l'activité devrait être plus faible.")


# Carte de Paris à 14h, le dimanche

st.header("Carte de Paris à 14h, le dimanche")

df_14h = df[(df['Heure'] == "14h") & (df["type_de_jour"] == "Jour de repos")]

trafic_comptage_14h = df_14h.groupby(['Nom du site de comptage', 'Coordonnées géographiques'])['Comptage horaire'].mean().reset_index()

top_sites = trafic_comptage_14h.nlargest(5, 'Comptage horaire')
bottom_sites = trafic_comptage_14h.nsmallest(5, 'Comptage horaire')

paris = folium.Map(location=[48.855578, 2.331830], zoom_start=12)

for index, site in trafic_comptage_14h.iterrows():
    latitude, longitude = map(float, site['Coordonnées géographiques'].split(','))
    taille_cercle = site['Comptage horaire'] / 10

    if site['Nom du site de comptage'] in top_sites['Nom du site de comptage'].tolist():
        couleur = 'green'
    elif site['Nom du site de comptage'] in bottom_sites['Nom du site de comptage'].tolist():
        couleur = 'red'
    else:
        couleur = 'blue'

    folium.CircleMarker(
        [latitude, longitude],
        radius=taille_cercle,
        popup=f"{site['Nom du site de comptage']} - {round(site['Comptage horaire'], 2)} vélos/heure",
        tooltip=site['Nom du site de comptage'],
        color=couleur,
        fill=True,
        fill_opacity=0.3
    ).add_to(paris)

folium_static(paris, width=700, height=500)

st.write("Cette fois-ci, la taille de nos cercles a drastiquement diminué, conformément à ce qui était attendu.")

st.write("Chose intéressante : nous avons une nouvelle forte zone d'affluence située entre les Invalides et le Pont Alexandre III, une zone très touristique de Paris. Nous voyons clairement que les Parisiens (ou des touristes en visite à Paris) aiment profiter de leur dimanche !")