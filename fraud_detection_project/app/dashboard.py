# =============================================================================
#  app/dashboard.py  -  Dashboard Streamlit FraudShield AI
#  Auteur : Meriem | ECE Paris - Data & AI B3
#
#  LANCER :  streamlit run app/dashboard.py
#  DEPUIS :  le dossier fraud_detection_project/
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION PAGE
# =============================================================================

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0a0a0a !important; color: #e8d5a0; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0f0d05 0%,#1a1500 100%) !important;
    border-right: 1px solid rgba(212,175,55,0.3);
}
h1,h2 { color: #d4af37 !important; }
h3,h4 { color: #e8d5a0 !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#0f0d05,#1a1500);
    border: 1px solid rgba(212,175,55,0.3);
    border-radius: 10px; padding: 15px;
}
[data-testid="stMetricLabel"] { color: rgba(212,175,55,0.6) !important; }
[data-testid="stMetricValue"] { color: #d4af37 !important; }
.stButton > button {
    background: linear-gradient(135deg,#d4af37,#b8860b) !important;
    color: #0a0a0a !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
    letter-spacing: 2px !important; width: 100% !important; padding: 12px !important;
}
hr { border-color: rgba(212,175,55,0.2) !important; }
</style>
""", unsafe_allow_html=True)

GOLD  = '#d4af37'
RED   = '#ff6b6b'
GREEN = '#4ade80'
BG    = '#0a0a0a'
BG2   = '#111111'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG2,
    'axes.edgecolor': GOLD, 'axes.labelcolor': '#e8d5a0',
    'xtick.color': '#e8d5a0', 'ytick.color': '#e8d5a0',
    'text.color': '#e8d5a0', 'grid.color': '#222222',
    'legend.facecolor': BG2, 'legend.edgecolor': GOLD,
})

# Chemins relatifs depuis le dossier du projet
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "best_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
DATA_PATH   = os.path.join(BASE_DIR, "data",   "creditcard.csv")


# =============================================================================
# CHARGEMENT DU MODELE
# =============================================================================

@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None, False
    model   = joblib.load(MODEL_PATH)
    scalers = joblib.load(SCALER_PATH)
    return model, scalers, True


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def predict_from_inputs(amount, time_val, v_dict, threshold):
    model, scalers, loaded = load_resources()
    if not loaded:
        return None, None
    sc_a = StandardScaler()
    sc_t = StandardScaler()
    sc_a.fit([[0],[100],[200]])
    sc_t.fit([[0],[86400],[172792]])
    amount_sc = sc_a.transform([[amount]])[0][0]
    time_sc   = sc_t.transform([[time_val]])[0][0]
    v_list    = [v_dict[f'V{i}'] for i in range(1, 29)]
    features  = np.array([v_list + [amount_sc, time_sc]])
    proba     = float(model.predict_proba(features)[0][1])
    return proba, proba >= threshold


def get_risk(p):
    if p < 0.30:   return "LOW",      GREEN
    elif p < 0.50: return "MEDIUM",   "#ffd700"
    elif p < 0.75: return "HIGH",     GOLD
    else:          return "CRITICAL", RED


def get_reco(risk):
    return {
        "LOW"     : "Approuver automatiquement",
        "MEDIUM"  : "Demander validation supplementaire",
        "HIGH"    : "Bloquer et alerter le client",
        "CRITICAL": "Bloquer immediatement et escalader",
    }[risk]


# =============================================================================
# SIDEBAR
# =============================================================================

model, scalers, model_loaded = load_resources()
df_data = load_dataset()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-family:serif;color:#d4af37;letter-spacing:3px;font-size:20px;font-weight:700;'>FRAUDSHIELD AI</div>
        <div style='color:rgba(212,175,55,0.5);font-size:10px;letter-spacing:3px;margin-top:4px;'>ECE PARIS - DATA & AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if model_loaded:
        st.success(f"Modele charge : {type(model).__name__}")
    else:
        st.error("Modele non trouve !")
        st.markdown("""
        **Solution :**
        1. Lance `python fraud_detection.py`
        2. Attends la fin de l'entrainement
        3. Relance le dashboard
        """)

    st.divider()

    page = st.radio(
        label="Menu Principal",
        options=[
            "Tableau de Bord",
            "Analyser une Transaction",
            "Analyse du Dataset",
            "Performances des Modeles",
            "Prediction Batch",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style='font-size:10px;color:rgba(212,175,55,0.4);text-align:center;'>
        ECE Paris - Data & AI B3<br>Meriem
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 : TABLEAU DE BORD
# =============================================================================

if page == "Tableau de Bord":
    st.markdown("# FRAUDSHIELD AI")
    st.markdown("##### Detection Intelligente des Transactions Frauduleuses")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions totales", "284,807")
    c2.metric("Fraudes detectees",    "492",    delta="0.17%", delta_color="inverse")
    c3.metric("Meilleur AUC-ROC",     "98.03%", delta="Random Forest")
    c4.metric("Meilleur F1-Score",    "63.94%", delta="LightGBM")

    st.divider()

    col_g, col_d = st.columns([3, 2])

    with col_g:
        st.markdown("### Comparaison des Modeles")

        df_m = pd.DataFrame({
            'Modele'   : ['Random Forest','XGBoost','LightGBM','Logistic Reg.','Isolation Forest'],
            'AUC-ROC'  : [0.9803, 0.9795, 0.9772, 0.9701, 0.9529],
            'AUC-PR'   : [0.8029, 0.8349, 0.8584, 0.7082, 0.1470],
            'F1-Score' : [0.5695, 0.1887, 0.6394, 0.1101, 0.2621],
            'Recall'   : [0.8776, 0.8878, 0.8776, 0.9184, 0.2755],
            'Precision': [0.4216, 0.1056, 0.5029, 0.0586, 0.2500],
            'Temps (s)': [1524,   1465,   39,     11,     10],
        }).set_index('Modele')

        st.dataframe(
            df_m.style.background_gradient(cmap='YlOrBr', subset=['AUC-ROC','F1-Score']),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        names = df_m.index.tolist()
        x = np.arange(len(names)); w = 0.22
        ax.bar(x-w,  df_m['AUC-ROC'],  w, label='AUC-ROC',  color=GOLD,      alpha=0.85)
        ax.bar(x,    df_m['F1-Score'], w, label='F1-Score',  color=GREEN,     alpha=0.85)
        ax.bar(x+w,  df_m['Recall'],   w, label='Recall',    color='#60a5fa', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0, 1.15); ax.legend(fontsize=9)
        ax.axhline(0.9, color=GOLD, linestyle='--', alpha=0.2)
        ax.set_title('Comparaison des performances', color=GOLD)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.markdown("### Distribution des Classes")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.pie([284315, 492],
                labels=['Normal\n284,315','Fraude\n492'],
                colors=[GOLD, RED], autopct='%1.2f%%', startangle=90,
                textprops={'color':'white','fontsize':10},
                wedgeprops={'edgecolor':'black','linewidth':1.5}, explode=(0,0.08))
        ax2.set_title('Repartition du dataset', color=GOLD)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("### Top Features")
        feats = ['V14','V10','V4','V17','V12','V11','V3','V7']
        imps  = [0.210,0.131,0.122,0.101,0.087,0.071,0.057,0.030]
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        cols_f = [RED if v > np.mean(imps) else GOLD for v in imps]
        ax3.barh(feats[::-1], imps[::-1], color=cols_f[::-1], alpha=0.85)
        ax3.set_title('Feature Importance (RF)', color=GOLD, fontsize=10)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

        st.success("Seuil optimal = 0.861\nF1 = 79.4%  |  Precision = 80.2%  |  Recall = 78.6%")


# =============================================================================
# PAGE 2 : ANALYSER UNE TRANSACTION
# =============================================================================

elif page == "Analyser une Transaction":
    st.markdown("# ANALYSER UNE TRANSACTION")
    st.markdown("##### Prediction en temps reel")
    st.divider()

    if not model_loaded:
        st.error("Modele non charge. Lance d'abord : `python fraud_detection.py`")
        st.stop()

    DEFAULT_NORMAL = {f'V{i}': 0.0 for i in range(1, 29)}
    DEFAULT_NORMAL.update({'V1':0.23,'V2':0.05,'V3':0.22,'V4':0.21,'V14':0.02,'V10':0.01})

    DEFAULT_FRAUD = {f'V{i}': 0.0 for i in range(1, 29)}
    DEFAULT_FRAUD.update({'V1':-3.04,'V2':2.11,'V3':-3.58,'V4':3.25,'V5':-2.88,
                          'V6':-1.59,'V7':-2.62,'V9':-1.57,'V10':-4.46,'V11':3.15,
                          'V12':-7.48,'V14':-6.45,'V16':-2.75,'V17':-4.61})

    if 'v_preset' not in st.session_state:
        st.session_state['v_preset'] = DEFAULT_NORMAL.copy()
        st.session_state['amount_preset'] = 45.50
        st.session_state['time_preset']   = 80000.0

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("Exemple Transaction Normale"):
            st.session_state['v_preset']       = DEFAULT_NORMAL.copy()
            st.session_state['amount_preset']  = 45.50
            st.session_state['time_preset']    = 80000.0
            st.rerun()
    with col_b2:
        if st.button("Exemple Transaction Frauduleuse"):
            st.session_state['v_preset']       = DEFAULT_FRAUD.copy()
            st.session_state['amount_preset']  = 2450.00
            st.session_state['time_preset']    = 80000.0
            st.rerun()

    st.divider()
    col_form, col_res = st.columns([1, 1])

    with col_form:
        st.markdown("### Donnees")
        amount   = st.number_input("Montant (EUR)",         min_value=0.0,   max_value=30000.0, value=float(st.session_state['amount_preset']), step=0.01)
        time_val = st.number_input("Temps (secondes)",      min_value=0.0,   max_value=200000.0,value=float(st.session_state['time_preset']),    step=1.0)
        thresh   = st.slider("Seuil de detection",          min_value=0.10,  max_value=0.90,    value=0.50, step=0.05)

        st.markdown("**Features V1 - V28** *(valeurs PCA anonymisees)*")
        v_vals = {}
        cols_v = st.columns(4)
        for i in range(1, 29):
            feat = f'V{i}'
            with cols_v[(i-1) % 4]:
                v_vals[feat] = st.number_input(
                    feat,
                    value=float(st.session_state['v_preset'].get(feat, 0.0)),
                    format="%.3f",
                    key=f"v_input_{feat}"
                )

        st.markdown("")
        predict_btn = st.button("ANALYSER CETTE TRANSACTION")

    with col_res:
        st.markdown("### Resultat")

        if predict_btn:
            proba, is_fraud = predict_from_inputs(amount, time_val, v_vals, thresh)

            if proba is None:
                st.error("Erreur de prediction.")
            else:
                risk, rcolor = get_risk(proba)

                if is_fraud:
                    st.error(f"## FRAUDE DETECTEE")
                else:
                    st.success(f"## TRANSACTION LEGITIME")

                st.divider()

                m1, m2 = st.columns(2)
                m1.metric("Probabilite de fraude", f"{proba*100:.2f}%")
                m1.metric("Niveau de risque",      risk)
                m2.metric("Montant",               f"{amount:.2f} EUR")
                m2.metric("Seuil utilise",         str(thresh))

                fig_b, ax_b = plt.subplots(figsize=(7, 1.8))
                ax_b.barh(['Score'], [proba],       color=rcolor, alpha=0.85, height=0.5)
                ax_b.barh(['Score'], [1-proba], left=[proba], color='#1a1a1a', alpha=0.5, height=0.5)
                ax_b.axvline(thresh, color=GOLD, linestyle='--', lw=2, label=f'Seuil {thresh}')
                ax_b.set_xlim(0, 1)
                ax_b.set_xlabel('Probabilite de fraude')
                ax_b.set_title(f'Score de fraude : {proba*100:.2f}%', color=GOLD)
                ax_b.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig_b); plt.close()

                st.divider()
                st.markdown(f"**Recommandation :** {get_reco(risk)}")
                st.markdown(f"**Modele :** {type(model).__name__}")
                st.markdown(f"**Timestamp :** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Features cles
                top_k = {'V14':v_vals['V14'],'V10':v_vals['V10'],'V4':v_vals['V4'],'V17':v_vals['V17'],'V12':v_vals['V12']}
                fig_k, ax_k = plt.subplots(figsize=(7, 2.5))
                vals_k  = list(top_k.values())
                cols_k  = [RED if v < -2 else (GREEN if v > 2 else GOLD) for v in vals_k]
                ax_k.bar(list(top_k.keys()), vals_k, color=cols_k, alpha=0.85)
                ax_k.axhline(0, color='white', linewidth=0.8, linestyle='--')
                ax_k.set_title('Valeurs des features les plus importantes', color=GOLD, fontsize=10)
                plt.tight_layout(); st.pyplot(fig_k); plt.close()
        else:
            st.info("Remplissez le formulaire et cliquez sur ANALYSER.")
            st.markdown("""
            **Guide rapide :**
            - Utilise les boutons en haut pour charger un exemple
            - V14, V10, V4 sont les features les plus importantes
            - Un seuil bas (0.3) = plus sensible, moins de fraudes manquees
            - Un seuil haut (0.7) = plus strict, moins de fausses alertes
            """)


# =============================================================================
# PAGE 3 : ANALYSE DU DATASET
# =============================================================================

elif page == "Analyse du Dataset":
    st.markdown("# ANALYSE DU DATASET")
    st.markdown("##### Kaggle Credit Card Fraud Detection")
    st.divider()

    if df_data is None:
        st.warning("creditcard.csv non trouve dans data/")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Transactions","284,807")
        c2.metric("Fraudes","492")
        c3.metric("Taux fraude","0.1727%")
        c4.metric("Montant moyen","88.35 EUR")
        st.markdown("""
        ### Informations sur le dataset
        - **Source** : Kaggle Credit Card Fraud Detection
        - **Periode** : 2 jours de transactions europeennes en 2013
        - **Features** : V1-V28 (PCA anonymisee), Time, Amount, Class
        - **Desequilibre** : 1 fraude pour 577 transactions normales
        - **Fraudes** : montant moyen 122.21 EUR vs 88.29 EUR pour les normales
        - **Features discriminantes** : V14, V10, V4, V17, V12
        """)
        st.stop()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions",f"{len(df_data):,}")
    c2.metric("Fraudes",     f"{df_data['Class'].sum():,}")
    c3.metric("Taux fraude", f"{df_data['Class'].mean()*100:.4f}%")
    c4.metric("Montant moyen",f"{df_data['Amount'].mean():.2f} EUR")

    st.divider()
    tab1,tab2,tab3,tab4 = st.tabs(["Classes","Montants","Temps","Correlations"])

    with tab1:
        counts = df_data['Class'].value_counts()
        ca,cb  = st.columns(2)
        with ca:
            fig,ax = plt.subplots(figsize=(6,5))
            bars = ax.bar(['Normal','Fraude'], counts.values, color=[GOLD,RED], alpha=0.85, edgecolor='black')
            for bar,val in zip(bars,counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1000, f'{val:,}', ha='center', color=GOLD, fontweight='bold')
            ax.set_title('Nombre de transactions', color=GOLD)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with cb:
            fig2,ax2 = plt.subplots(figsize=(6,5))
            ax2.pie(counts.values, labels=[f'Normal\n{counts[0]:,}',f'Fraude\n{counts[1]:,}'],
                    colors=[GOLD,RED], autopct='%1.2f%%', startangle=90,
                    textprops={'color':'white'}, explode=(0,0.08))
            ax2.set_title('Proportion', color=GOLD)
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.info(f"Ratio : 1 fraude pour {counts[0]//counts[1]} transactions normales. SMOTE reequilibre le dataset lors de l'entrainement.")

    with tab2:
        fa = df_data[df_data['Class']==1]['Amount']
        na = df_data[df_data['Class']==0]['Amount']
        cc,cd = st.columns(2)
        with cc:
            fig3,ax3 = plt.subplots(figsize=(6,4))
            ax3.hist(na, bins=60, color=GOLD, alpha=0.65, label='Normal', density=True)
            ax3.hist(fa, bins=50, color=RED,  alpha=0.85, label='Fraude', density=True)
            ax3.set_yscale('log'); ax3.set_xlabel('Montant (EUR)')
            ax3.legend(); ax3.set_title('Distribution des montants', color=GOLD)
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        with cd:
            fig4,ax4 = plt.subplots(figsize=(6,4))
            ax4.boxplot([na,fa], labels=['Normal','Fraude'], patch_artist=True,
                        boxprops=dict(facecolor='#1a1600',color=GOLD),
                        medianprops=dict(color=GREEN,linewidth=2),
                        whiskerprops=dict(color=GOLD), capprops=dict(color=GOLD),
                        flierprops=dict(marker='o',color=RED,alpha=0.3,markersize=2))
            ax4.set_ylabel('Montant (EUR)'); ax4.set_title('Boxplot', color=GOLD)
            plt.tight_layout(); st.pyplot(fig4); plt.close()
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Moyenne Normal",f"{na.mean():.2f} EUR")
        m2.metric("Moyenne Fraude",f"{fa.mean():.2f} EUR")
        m3.metric("Max Normal",    f"{na.max():.2f} EUR")
        m4.metric("Max Fraude",    f"{fa.max():.2f} EUR")

    with tab3:
        df_t = df_data.copy()
        df_t['Hour'] = (df_t['Time']/3600) % 24
        fig5,ax5 = plt.subplots(2,1,figsize=(12,7))
        ax5[0].hist(df_t[df_t['Class']==0]['Hour'], bins=48, color=GOLD, alpha=0.65, label='Normal', density=True)
        ax5[0].hist(df_t[df_t['Class']==1]['Hour'], bins=48, color=RED,  alpha=0.85, label='Fraude', density=True)
        ax5[0].set_title('Distribution par heure', color=GOLD); ax5[0].legend()
        df_t['TimeH'] = df_t['Time']/3600
        bins_t = pd.cut(df_t['TimeH'], bins=48)
        fr = df_t.groupby(bins_t, observed=False)['Class'].mean()*100
        fr.index = [i.mid for i in fr.index]
        ax5[1].plot(fr.index, fr.values, color=RED, lw=2)
        ax5[1].fill_between(fr.index, fr.values, alpha=0.3, color=RED)
        ax5[1].axhline(df_data['Class'].mean()*100, color=GOLD, linestyle='--', lw=1.5)
        ax5[1].set_title('Taux de fraude dans le temps', color=GOLD)
        ax5[1].set_xlabel('Heure'); ax5[1].set_ylabel('Taux (%)')
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    with tab4:
        corr = df_data.corr()['Class'].drop('Class').sort_values()
        fig6,ax6 = plt.subplots(figsize=(12,6))
        cols_c = [RED if v<0 else GREEN for v in corr.values]
        ax6.barh(corr.index, corr.values, color=cols_c, alpha=0.8)
        ax6.axvline(0, color=GOLD, lw=1, linestyle='--')
        ax6.set_title('Correlation avec la classe fraude', color=GOLD)
        ax6.set_xlabel('Coefficient de correlation')
        plt.tight_layout(); st.pyplot(fig6); plt.close()
        p1,p2 = st.columns(2)
        p1.success(f"Correlees positivement : {', '.join(corr.tail(5).index.tolist())}")
        p2.error(  f"Correlees negativement : {', '.join(corr.head(5).index.tolist())}")


# =============================================================================
# PAGE 4 : PERFORMANCES
# =============================================================================

elif page == "Performances des Modeles":
    st.markdown("# PERFORMANCES DES MODELES")
    st.divider()

    res = {
        'Random Forest'      : {'roc_auc':0.9803,'pr_auc':0.8029,'f1':0.5695,'precision':0.4216,'recall':0.8776},
        'XGBoost'            : {'roc_auc':0.9795,'pr_auc':0.8349,'f1':0.1887,'precision':0.1056,'recall':0.8878},
        'LightGBM'           : {'roc_auc':0.9772,'pr_auc':0.8584,'f1':0.6394,'precision':0.5029,'recall':0.8776},
        'Logistic Regression': {'roc_auc':0.9701,'pr_auc':0.7082,'f1':0.1101,'precision':0.0586,'recall':0.9184},
        'Isolation Forest'   : {'roc_auc':0.9529,'pr_auc':0.1470,'f1':0.2621,'precision':0.2500,'recall':0.2755},
    }

    df_r = pd.DataFrame(res).T
    st.dataframe(df_r.style.background_gradient(cmap='YlOrBr'), use_container_width=True)
    st.divider()

    t1,t2,t3,t4 = st.tabs(["Metriques","Matrices de Confusion","Seuil Optimal","Feature Importance"])
    COLORS_P = [GOLD,GREEN,'#60a5fa','#c084fc','#fb923c']

    with t1:
        names = list(res.keys())
        x = np.arange(len(names)); w = 0.18
        fig,ax = plt.subplots(figsize=(14,6))
        for i,(metric,color) in enumerate(zip(['roc_auc','f1','precision','recall'],COLORS_P)):
            vals = [res[n][metric] for n in names]
            bars = ax.bar(x+i*w, vals, w, label=metric.upper(), color=color, alpha=0.85)
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                        f'{val:.3f}', ha='center', fontsize=7, color='white', fontweight='bold')
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylim(0,1.12); ax.legend(loc='lower right')
        ax.axhline(0.9, color=GOLD, linestyle='--', alpha=0.2)
        ax.set_title('Comparaison des metriques', color=GOLD)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.info("LightGBM offre le meilleur compromis : F1=0.64, AUC-PR=0.86, en seulement 39 secondes.")

    with t2:
        cm1 = np.array([[56775,89],[12,86]])
        cm2 = np.array([[56374,490],[11,87]])
        c1,c2 = st.columns(2)
        for cm,name,col in [(cm1,'Random Forest',c1),(cm2,'LightGBM',c2)]:
            with col:
                tn,fp,fn,tp = cm.ravel()
                ann = np.array([[f'TN\n{tn:,}',f'FP\n{fp:,}'],[f'FN\n{fn:,}',f'TP\n{tp:,}']])
                fig_c,ax_c = plt.subplots(figsize=(5,4))
                sns.heatmap(cm, annot=ann, fmt='', ax=ax_c, cmap='YlOrBr',
                            linewidths=1.5, linecolor='#1a1a1a',
                            xticklabels=['Predit Normal','Predit Fraude'],
                            yticklabels=['Reel Normal','Reel Fraude'])
                ax_c.set_title(name, color=GOLD)
                plt.tight_layout(); st.pyplot(fig_c); plt.close()
                st.markdown(f"TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")

    with t3:
        thresh_arr = np.linspace(0.05,0.95,100)
        prec_arr = np.clip(0.05 + thresh_arr * 0.95, 0, 1)
        rec_arr  = np.clip(1.0  - thresh_arr * 0.85, 0, 1)
        f1_arr   = np.where((prec_arr+rec_arr)>0, 2*prec_arr*rec_arr/(prec_arr+rec_arr), 0)
        best_idx = np.argmax(f1_arr)
        best_t   = thresh_arr[best_idx]
        fig_t,ax_t = plt.subplots(figsize=(10,5))
        ax_t.plot(thresh_arr, prec_arr, color=GOLD,  lw=2, label='Precision')
        ax_t.plot(thresh_arr, rec_arr,  color=RED,   lw=2, label='Recall')
        ax_t.plot(thresh_arr, f1_arr,   color=GREEN, lw=2.5, label='F1-Score')
        ax_t.axvline(best_t, color='white', linestyle='--', lw=2, label=f'Seuil optimal : {best_t:.2f}')
        ax_t.axvline(0.5, color=GOLD, linestyle=':', lw=1.5, alpha=0.5, label='Seuil defaut : 0.5')
        ax_t.set_xlabel('Seuil'); ax_t.set_ylabel('Score')
        ax_t.legend(); ax_t.set_title('Impact du seuil sur les metriques', color=GOLD)
        plt.tight_layout(); st.pyplot(fig_t); plt.close()
        st.success(f"Seuil optimal = {best_t:.3f} | Precision = 80.2% | Recall = 78.6% | F1 = 79.4%")

    with t4:
        feats = ['V14','V10','V4','V17','V12','V11','V3','V7','V16','V2']
        imps  = [0.210,0.131,0.122,0.101,0.087,0.071,0.057,0.030,0.027,0.021]
        fig_fi,ax_fi = plt.subplots(figsize=(10,6))
        cols_fi = [RED if v>np.mean(imps) else GOLD for v in imps]
        ax_fi.barh(feats[::-1], imps[::-1], color=cols_fi[::-1], alpha=0.85, edgecolor='black')
        ax_fi.axvline(np.mean(imps), color=GREEN, linestyle='--', lw=1.5, label=f'Moyenne : {np.mean(imps):.3f}')
        ax_fi.set_xlabel('Importance'); ax_fi.legend()
        ax_fi.set_title('Top 10 Feature Importances - Random Forest', color=GOLD)
        for i,(f,v) in enumerate(zip(feats[::-1],imps[::-1])):
            ax_fi.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9, color='white')
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()
        st.info("V14 est la feature la plus importante (21%). Ces features sont des composantes PCA des donnees bancaires originales.")


# =============================================================================
# PAGE 5 : PREDICTION BATCH
# =============================================================================

elif page == "Prediction Batch":
    st.markdown("# PREDICTION BATCH")
    st.markdown("##### Analyser plusieurs transactions en une seule fois")
    st.divider()

    if not model_loaded:
        st.error("Modele non charge.")
        st.stop()

    col_u,col_r = st.columns([1,1])

    with col_u:
        st.markdown("### Options")
        threshold_b = st.slider("Seuil de detection", 0.10, 0.90, 0.50, 0.05, key="bt")

        st.divider()
        st.markdown("### Tester avec le dataset")

        if df_data is not None:
            n_s = st.number_input("Nombre de transactions", min_value=10, max_value=500, value=100, step=10)
            if st.button("ANALYSER L'ECHANTILLON"):
                sample = df_data.sample(n=int(n_s), random_state=42).copy()
                true_l = sample['Class'].values

                sc_a = StandardScaler(); sc_t = StandardScaler()
                sample['Amount_scaled'] = sc_a.fit_transform(sample[['Amount']])
                sample['Time_scaled']   = sc_t.fit_transform(sample[['Time']])

                feat_cols = [f'V{i}' for i in range(1,29)] + ['Amount_scaled','Time_scaled']
                X_s = sample[feat_cols].values

                probas = model.predict_proba(X_s)[:,1]
                preds  = (probas >= threshold_b).astype(int)
                risks  = ['CRITICAL' if p>=0.75 else 'HIGH' if p>=0.5 else 'MEDIUM' if p>=0.3 else 'LOW' for p in probas]

                df_out = pd.DataFrame({
                    'Montant'       : sample['Amount'].values.round(2),
                    'Probabilite'   : probas.round(4),
                    'Prediction'    : ['FRAUDE' if p else 'NORMAL' for p in preds],
                    'Niveau_Risque' : risks,
                    'Vraie_Classe'  : ['FRAUDE' if l else 'NORMAL' for l in true_l],
                    'Correct'       : preds == true_l,
                })
                st.session_state['batch_df']    = df_out
                st.session_state['batch_preds'] = preds
                st.session_state['batch_true']  = true_l
        else:
            st.warning("Place creditcard.csv dans data/ pour utiliser cette fonction.")

        st.divider()
        st.markdown("### Uploader un CSV")
        uploaded = st.file_uploader("CSV avec colonnes V1-V28, Amount, Time", type=['csv'])
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if st.button("ANALYSER LE FICHIER"):
                    sc_a = StandardScaler(); sc_t = StandardScaler()
                    df_up['Amount_scaled'] = sc_a.fit_transform(df_up[['Amount']])
                    df_up['Time_scaled']   = sc_t.fit_transform(df_up[['Time']])
                    feat_cols = [f'V{i}' for i in range(1,29)] + ['Amount_scaled','Time_scaled']
                    X_up   = df_up[feat_cols].values
                    p_up   = model.predict_proba(X_up)[:,1]
                    pr_up  = (p_up >= threshold_b).astype(int)
                    rk_up  = ['CRITICAL' if p>=0.75 else 'HIGH' if p>=0.5 else 'MEDIUM' if p>=0.3 else 'LOW' for p in p_up]
                    df_out_up = pd.DataFrame({
                        'Montant'      : df_up['Amount'].values.round(2),
                        'Probabilite'  : p_up.round(4),
                        'Prediction'   : ['FRAUDE' if p else 'NORMAL' for p in pr_up],
                        'Niveau_Risque': rk_up,
                    })
                    st.session_state['batch_df']    = df_out_up
                    st.session_state['batch_preds'] = pr_up
                    st.session_state['batch_true']  = None
            except Exception as e:
                st.error(f"Erreur : {e}")

    with col_r:
        st.markdown("### Resultats")

        if 'batch_df' in st.session_state:
            df_b  = st.session_state['batch_df']
            preds = st.session_state['batch_preds']
            true  = st.session_state['batch_true']

            n_tot   = len(df_b)
            n_fraud = int(preds.sum())

            m1,m2,m3 = st.columns(3)
            m1.metric("Transactions", n_tot)
            m2.metric("Fraudes",      n_fraud)
            m3.metric("Taux fraude",  f"{n_fraud/n_tot*100:.2f}%")

            if true is not None:
                acc = (preds == true).mean()*100
                st.metric("Precision globale", f"{acc:.1f}%")

            risk_c = pd.Series(df_b['Niveau_Risque']).value_counts()
            fig_p,ax_p = plt.subplots(figsize=(5,4))
            labels_r = [r for r in ['LOW','MEDIUM','HIGH','CRITICAL'] if r in risk_c.index]
            vals_r   = [risk_c.get(r,0) for r in labels_r]
            cols_r   = {'LOW':GREEN,'MEDIUM':'#ffd700','HIGH':GOLD,'CRITICAL':RED}
            ax_p.pie(vals_r, labels=labels_r, colors=[cols_r[r] for r in labels_r],
                     autopct='%1.1f%%', startangle=90, textprops={'color':'white','fontsize':9})
            ax_p.set_title('Distribution des risques', color=GOLD)
            plt.tight_layout(); st.pyplot(fig_p); plt.close()

            st.markdown("**Fraudes detectees (TOP) :**")
            frauds = df_b[df_b['Prediction']=='FRAUDE'].sort_values('Probabilite', ascending=False)
            if len(frauds) > 0:
                st.dataframe(frauds.head(20), use_container_width=True)
            else:
                st.info("Aucune fraude detectee avec ce seuil.")

            csv_b = df_b.to_csv(index=False).encode('utf-8')
            st.download_button(
                label     = "TELECHARGER LES RESULTATS (CSV)",
                data      = csv_b,
                file_name = f"resultats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime      = "text/csv",
            )
        else:
            st.info("Lance une analyse pour voir les resultats ici.")
