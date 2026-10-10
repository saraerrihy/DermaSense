# -*- coding: utf-8 -*-
"""
================================================================================
 PROJET DERMASENSE — PFA IAD
 Recommandation de soins cosmétiques personnalisés par Intelligence Artificielle
--------------------------------------------------------------------------------
 Pipeline complet et reproductible :
   1. Chargement et analyse exploratoire (EDA) du dataset Cosmetics (Sephora)
   2. Simulation documentée d'un dataset clients (sexe, âge, type de peau)
   3. Prétraitement (nettoyage, TF-IDF des ingrédients, encodage, normalisation)
   4. Modèle A : prédiction de la compatibilité "peau sensible" (données réelles)
   5. Modèle B : prédiction de la satisfaction client (interactions simulées)
   6. Évaluation comparative + analyse par sous-groupes (sexe, âge) + biais
--------------------------------------------------------------------------------
 Auteur : Maryam Farhoune — 4A IAD | Reproductibilité : seed = 42
 Prérequis : pip install pandas numpy scikit-learn matplotlib seaborn
 NB : les métriques du rapport ont été produites avec ce même protocole
 (mêmes données, mêmes seeds) ; de légères variations (±0,01-0,03) sont
 normales selon les versions de scikit-learn.
================================================================================
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # rendu sans écran (serveur)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", palette="rocket")
PALETTE = ["#7B2D62", "#C0567E", "#E89A8D", "#F2C57C", "#5C7457"]
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
RESULTS = {}

# ==============================================================================
# SECTION 1 — CHARGEMENT ET ANALYSE EXPLORATOIRE (EDA)
# ==============================================================================
df = pd.read_csv("cosmetics.csv")
print("Dimensions :", df.shape)
print(df.dtypes)
print("Valeurs manquantes :\n", df.isna().sum())
print("Doublons :", df.duplicated().sum())

RESULTS["dataset"] = {
    "lignes": int(df.shape[0]), "colonnes": int(df.shape[1]),
    "manquants": int(df.isna().sum().sum()),
    "doublons": int(df.duplicated().sum()),
    "marques": int(df["Brand"].nunique()),
    "prix_moyen": round(float(df["Price"].mean()), 2),
    "prix_median": round(float(df["Price"].median()), 2),
    "rank_moyen": round(float(df["Rank"].mean()), 2),
    "categories": df["Label"].value_counts().to_dict(),
}

# --- Fig 1 : distribution des catégories de produits
plt.figure(figsize=(8, 4.5))
ax = df["Label"].value_counts().plot(kind="bar", color=PALETTE[0])
ax.set_title("Répartition des produits par catégorie")
ax.set_xlabel("Catégorie"); ax.set_ylabel("Nombre de produits")
plt.xticks(rotation=30); plt.tight_layout()
plt.savefig(f"{FIG}/fig1_categories.png", dpi=150); plt.close()

# --- Fig 2 : distribution des prix (histogramme) + boxplot par catégorie
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(df["Price"], bins=40, color=PALETTE[1], edgecolor="white")
axes[0].set_title("Distribution des prix (USD)")
axes[0].set_xlabel("Prix"); axes[0].set_ylabel("Effectif")
sns.boxplot(data=df, x="Label", y="Price", ax=axes[1], palette=PALETTE)
axes[1].set_title("Prix par catégorie (valeurs aberrantes visibles)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_prix.png", dpi=150); plt.close()

# Détection d'aberrations (règle IQR sur le prix)
q1, q3 = df["Price"].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[df["Price"] > q3 + 1.5 * iqr]
RESULTS["dataset"]["outliers_prix"] = int(len(outliers))
RESULTS["dataset"]["seuil_outlier"] = round(float(q3 + 1.5 * iqr), 1)

# --- Fig 3 : heatmap de corrélations (variables numériques)
plt.figure(figsize=(7.5, 6))
num_cols = ["Price", "Rank", "Combination", "Dry", "Normal", "Oily", "Sensitive"]
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="rocket_r",
            linewidths=0.5)
plt.title("Matrice de corrélations")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_correlations.png", dpi=150); plt.close()

# --- Fig 4 : distribution du Rank
plt.figure(figsize=(7, 4))
plt.hist(df["Rank"], bins=30, color=PALETTE[3], edgecolor="white")
plt.axvline(df["Rank"].median(), color=PALETTE[0], ls="--",
            label=f"Médiane = {df['Rank'].median():.1f}")
plt.title("Distribution des notes (Rank)"); plt.legend()
plt.xlabel("Note moyenne"); plt.ylabel("Effectif")
plt.tight_layout(); plt.savefig(f"{FIG}/fig4_rank.png", dpi=150); plt.close()

# ==============================================================================
# SECTION 2 — SIMULATION DOCUMENTÉE DU DATASET CLIENTS
# ------------------------------------------------------------------------------
# Le dataset Cosmetics décrit des PRODUITS ; les variables sexe et âge sont
# collectées par DermaSense à l'inscription (signup). Pour entraîner et auditer
# le moteur de recommandation, nous simulons une base clients réaliste fondée
# sur les statistiques du marché cosmétique marocain (≈72% de clientèle
# féminine, âge concentré entre 18 et 45 ans).
# ==============================================================================
N_CLIENTS = 3000
sexe = np.random.choice(["Femme", "Homme"], N_CLIENTS, p=[0.72, 0.28])
age = np.clip(np.random.normal(31, 11, N_CLIENTS).round(), 16, 70).astype(int)
peau = np.random.choice(["Combination", "Dry", "Normal", "Oily", "Sensitive"],
                        N_CLIENTS, p=[0.26, 0.20, 0.22, 0.18, 0.14])
budget = np.clip(np.random.lognormal(3.6, 0.55, N_CLIENTS), 8, 300).round(0)

clients = pd.DataFrame({"Sexe": sexe, "Age": age, "TypePeau": peau,
                        "Budget": budget})
clients["TrancheAge"] = pd.cut(clients["Age"], [15, 25, 35, 50, 70],
                               labels=["16-25", "26-35", "36-50", "51-70"])
clients.to_csv("clients_simules.csv", index=False)

RESULTS["clients"] = {
    "n": N_CLIENTS,
    "femmes_pct": round(float((clients.Sexe == "Femme").mean()) * 100, 1),
    "age_moyen": round(float(clients.Age.mean()), 1),
    "tranches": clients["TrancheAge"].value_counts().sort_index().astype(int).to_dict(),
}

# --- Fig 5 : profils clients — sexe et âge
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
clients["Sexe"].value_counts().plot(kind="pie", autopct="%1.1f%%",
                                    colors=[PALETTE[0], PALETTE[3]],
                                    ax=axes[0], ylabel="")
axes[0].set_title("Répartition par sexe (base clients)")
sns.histplot(data=clients, x="Age", hue="Sexe", multiple="stack",
             palette=[PALETTE[0], PALETTE[3]], bins=25, ax=axes[1])
axes[1].set_title("Distribution des âges par sexe")
plt.tight_layout(); plt.savefig(f"{FIG}/fig5_clients.png", dpi=150); plt.close()

# --- Interactions client-produit : chaque client évalue 4 produits proposés
rows = []
prod = df.reset_index(drop=True)
for i, c in clients.iterrows():
    idx = np.random.choice(len(prod), 4, replace=False)
    for j in idx:
        p = prod.iloc[j]
        compat = int(p[c["TypePeau"]])           # produit adapté au type de peau
        abordable = int(p["Price"] <= c["Budget"] * 1.2)
        score = (2.2 * compat + 1.4 * abordable + 0.9 * (p["Rank"] >= 4.2)
                 + np.random.normal(0, 0.9))
        # légère hétérogénéité comportementale par âge (tolérance prix)
        if c["Age"] > 50 and p["Price"] > 100:
            score -= 0.4
        rows.append({"Sexe": c["Sexe"], "Age": c["Age"],
                     "TrancheAge": str(c["TrancheAge"]), "Budget": c["Budget"],
                     "TypePeau": c["TypePeau"], "Categorie": p["Label"],
                     "Prix": p["Price"], "Rank": p["Rank"], "Compat": compat,
                     "Satisfait": int(score > 2.4)})
inter = pd.DataFrame(rows)
inter.to_csv("interactions_simulees.csv", index=False)
RESULTS["interactions"] = {"n": len(inter),
                           "taux_satisfaction": round(float(inter.Satisfait.mean()) * 100, 1)}

# ==============================================================================
# SECTION 3 — MODÈLE A : COMPATIBILITÉ PEAU SENSIBLE (DONNÉES RÉELLES)
# ------------------------------------------------------------------------------
# Objectif : prédire, à partir de la composition (ingrédients) et des attributs
# du produit, s'il convient aux peaux sensibles. C'est le cœur scientifique du
# moteur DermaSense : étendre la compatibilité aux produits non étiquetés.
# ==============================================================================
dfA = df.copy()
dfA["Ingredients"] = dfA["Ingredients"].str.lower()
yA = dfA["Sensitive"]
RESULTS["modeleA_cible"] = {"positifs": int(yA.sum()),
                            "negatifs": int((1 - yA).sum()),
                            "taux_positif": round(float(yA.mean()) * 100, 1)}

preproc_A = ColumnTransformer([
    ("tfidf", TfidfVectorizer(max_features=300, token_pattern=r"[a-z][a-z\- ]+",
                              stop_words=None), "Ingredients"),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["Label"]),
    ("num", StandardScaler(), ["Price", "Rank"]),
])

XA_tr, XA_te, yA_tr, yA_te = train_test_split(
    dfA[["Ingredients", "Label", "Price", "Rank"]], yA,
    test_size=0.2, random_state=SEED, stratify=yA)

models = {
    "Régression Logistique": LogisticRegression(max_iter=2000, C=1.0,
                                                class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=None,
                                            class_weight="balanced",
                                            random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=300,
                                                    learning_rate=0.08,
                                                    max_depth=3,
                                                    random_state=SEED),
}

def evaluate(name, pipe, Xtr, ytr, Xte, yte):
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)
    proba = pipe.predict_proba(Xte)[:, 1]
    return {
        "accuracy": round(accuracy_score(yte, yp), 3),
        "precision": round(precision_score(yte, yp), 3),
        "rappel": round(recall_score(yte, yp), 3),
        "f1": round(f1_score(yte, yp), 3),
        "auc": round(roc_auc_score(yte, proba), 3),
    }, yp, proba, pipe

resA, fitted_A = {}, {}
for name, mdl in models.items():
    pipe = Pipeline([("prep", preproc_A), ("clf", mdl)])
    resA[name], yp, proba, fitted = evaluate(name, pipe, XA_tr, yA_tr, XA_te, yA_te)
    fitted_A[name] = (fitted, yp, proba)
RESULTS["modeleA"] = resA
bestA = max(resA, key=lambda k: resA[k]["f1"])
RESULTS["modeleA_best"] = bestA
print("\n=== MODÈLE A — Compatibilité peau sensible ===")
print(pd.DataFrame(resA).T)

# --- Fig 6 : matrice de confusion du meilleur modèle A + courbes ROC
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cm = confusion_matrix(yA_te, fitted_A[bestA][1])
sns.heatmap(cm, annot=True, fmt="d", cmap="rocket_r", ax=axes[0],
            xticklabels=["Non adapté", "Adapté"],
            yticklabels=["Non adapté", "Adapté"])
axes[0].set_title(f"Matrice de confusion — {bestA}")
axes[0].set_xlabel("Prédit"); axes[0].set_ylabel("Réel")
for name, (f, yp, proba) in fitted_A.items():
    fpr, tpr, _ = roc_curve(yA_te, proba)
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={resA[name]['auc']})")
axes[1].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[1].set_title("Courbes ROC — Modèle A"); axes[1].legend()
axes[1].set_xlabel("Taux de faux positifs"); axes[1].set_ylabel("Taux de vrais positifs")
plt.tight_layout(); plt.savefig(f"{FIG}/fig6_modeleA.png", dpi=150); plt.close()
RESULTS["modeleA_cm"] = cm.tolist()

# ==============================================================================
# SECTION 4 — MODÈLE B : SATISFACTION CLIENT (SEXE ET ÂGE AU CŒUR DU MODÈLE)
# ==============================================================================
featB_cat = ["Sexe", "TypePeau", "Categorie"]
featB_num = ["Age", "Budget", "Prix", "Rank", "Compat"]
XB = inter[featB_cat + featB_num]
yB = inter["Satisfait"]

preproc_B = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), featB_cat),
    ("num", StandardScaler(), featB_num),
])
XB_tr, XB_te, yB_tr, yB_te = train_test_split(XB, yB, test_size=0.2,
                                              random_state=SEED, stratify=yB)
te_meta = inter.loc[XB_te.index, ["Sexe", "TrancheAge"]]

resB, fitted_B = {}, {}
for name, mdl in models.items():
    pipe = Pipeline([("prep", preproc_B), ("clf", mdl)])
    resB[name], yp, proba, fitted = evaluate(name, pipe, XB_tr, yB_tr, XB_te, yB_te)
    fitted_B[name] = (fitted, yp, proba)
RESULTS["modeleB"] = resB
bestB = max(resB, key=lambda k: resB[k]["f1"])
RESULTS["modeleB_best"] = bestB
print("\n=== MODÈLE B — Satisfaction client ===")
print(pd.DataFrame(resB).T)

# --- Fig 7 : matrice de confusion modèle B + comparaison F1 des 6 modèles
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cmB = confusion_matrix(yB_te, fitted_B[bestB][1])
sns.heatmap(cmB, annot=True, fmt="d", cmap="rocket_r", ax=axes[0],
            xticklabels=["Insatisfait", "Satisfait"],
            yticklabels=["Insatisfait", "Satisfait"])
axes[0].set_title(f"Matrice de confusion — {bestB}")
axes[0].set_xlabel("Prédit"); axes[0].set_ylabel("Réel")
comp = pd.DataFrame({
    "Modèle A (peau sensible)": {k: v["f1"] for k, v in resA.items()},
    "Modèle B (satisfaction)": {k: v["f1"] for k, v in resB.items()},
})
comp.plot(kind="bar", ax=axes[1], color=[PALETTE[0], PALETTE[3]])
axes[1].set_title("Comparaison des F1-scores"); axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=15)
plt.tight_layout(); plt.savefig(f"{FIG}/fig7_modeleB.png", dpi=150); plt.close()
RESULTS["modeleB_cm"] = cmB.tolist()

# ==============================================================================
# SECTION 5 — ANALYSE PAR SOUS-GROUPES (FEMMES/HOMMES, TRANCHES D'ÂGE) & BIAIS
# ==============================================================================
best_pipe, ypB, probaB = fitted_B[bestB]
sub = te_meta.copy()
sub["y"] = yB_te.values; sub["yp"] = ypB

def group_metrics(frame, col):
    out = {}
    for g, gdf in frame.groupby(col):
        if len(gdf) < 30:
            continue
        out[str(g)] = {
            "n": int(len(gdf)),
            "accuracy": round(accuracy_score(gdf.y, gdf.yp), 3),
            "f1": round(f1_score(gdf.y, gdf.yp), 3),
            "rappel": round(recall_score(gdf.y, gdf.yp), 3),
            "taux_satisfaction_reel": round(float(gdf.y.mean()), 3),
        }
    return out

RESULTS["sous_groupes_sexe"] = group_metrics(sub, "Sexe")
RESULTS["sous_groupes_age"] = group_metrics(sub, "TrancheAge")
print("\n=== Sous-groupes (sexe) ===\n", pd.DataFrame(RESULTS["sous_groupes_sexe"]).T)
print("\n=== Sous-groupes (âge) ===\n", pd.DataFrame(RESULTS["sous_groupes_age"]).T)

# --- Fig 8 : performances par sous-groupe
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
pd.DataFrame(RESULTS["sous_groupes_sexe"]).T[["accuracy", "f1"]].plot(
    kind="bar", ax=axes[0], color=[PALETTE[0], PALETTE[3]])
axes[0].set_title("Performance par sexe"); axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=0)
ordre = ["16-25", "26-35", "36-50", "51-70"]
dfage = pd.DataFrame(RESULTS["sous_groupes_age"]).T.reindex(ordre)
dfage[["accuracy", "f1"]].plot(kind="bar", ax=axes[1],
                               color=[PALETTE[0], PALETTE[3]])
axes[1].set_title("Performance par tranche d'âge"); axes[1].set_ylim(0, 1)
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(f"{FIG}/fig8_sousgroupes.png", dpi=150); plt.close()

# --- Importance des variables (Random Forest, modèle B)
rf_pipe = fitted_B["Random Forest"][0]
feat_names = rf_pipe.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_pipe.named_steps["clf"].feature_importances_,
                index=feat_names).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 4.5))
imp[::-1].plot(kind="barh", color=PALETTE[0])
plt.title("Top 10 variables — importance (Random Forest, modèle B)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig9_importance.png", dpi=150); plt.close()
RESULTS["importance_top5"] = {k: round(float(v), 3) for k, v in imp.head(5).items()}

# ==============================================================================
# SECTION 6 — RECOMMANDATION PAR SIMILARITÉ D'INGRÉDIENTS (NLP)
# ------------------------------------------------------------------------------
# Vectorisation TF-IDF de la liste complète des ingrédients (tokens = segments
# séparés par des virgules) puis similarité cosinus entre produits. Usages :
# proposer des alternatives à formulation proche (« dupes »), y compris moins
# chères, et recommander un produit jamais noté (démarrage à froid).
# ==============================================================================
from sklearn.metrics.pairwise import cosine_similarity

tfidf_sim = TfidfVectorizer(token_pattern=r"[^,]+", lowercase=True, min_df=2)
M = tfidf_sim.fit_transform(df["Ingredients"].str.lower())
SIM = cosine_similarity(M)

def recommander_similaires(nom_produit, k=5):
    """Top-k produits à la composition la plus proche (similarité cosinus)."""
    idx = df.index[df["Name"] == nom_produit][0]
    ordre = SIM[idx].argsort()[::-1]
    voisins = [j for j in ordre if j != idx][:k]
    return df.loc[voisins, ["Name", "Brand", "Label", "Price", "Rank"]].assign(
        Similarite=[round(float(SIM[idx, j]), 3) for j in voisins])

exemple = recommander_similaires("Crème de la Mer", 5)
print("\n=== Similarité d'ingrédients — exemple « Crème de la Mer » ===")
print(exemple.to_string(index=False))
RESULTS["similarite_exemple"] = exemple.to_dict("records")

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, ensure_ascii=False, indent=2, default=str)
print("\nPipeline terminé — figures dans ./figures, métriques dans results.json")
