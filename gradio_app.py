import warnings
import os

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from gradio import Blocks, Column, Dropdown, Markdown, Number, Plot, Row, Slider
import gradio as gr
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

DATA_PATH = "seer_cancer.csv"

RAW_FEATURES = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Reginol Node Positive",
    "T Stage",
    "N Stage",
    "Grade",
    "A Stage",
    "Estrogen Status",
    "Progesterone Status",
]


class RSFSurvivalService:
    def __init__(self) -> None:
        self.df = self._load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._build_train_test(self.df)
        self.feature_columns = self.X_train.columns.tolist()

        self.rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_split=10,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        self.rsf.fit(self.X_train, self.y_train)

        background = self.X_train.sample(n=min(200, len(self.X_train)), random_state=42)
        self.shap_explainer = shap.Explainer(self.rsf.predict, background)

        self.defaults_num = {
            "Regional Node Examined": float(self.df["Regional Node Examined"].median()),
            "Reginol Node Positive": float(self.df["Reginol Node Positive"].median()),
            "N Stage": self.df["N Stage"].mode().iloc[0],
            "A Stage": self.df["A Stage"].mode().iloc[0],
        }

        self.choices = {
            "T Stage": sorted(self.df["T Stage"].dropna().astype(str).unique().tolist()),
            "N Stage": sorted(self.df["N Stage"].dropna().astype(str).unique().tolist()),
            "Grade": sorted(self.df["Grade"].dropna().astype(str).unique().tolist()),
            "Estrogen Status": sorted(self.df["Estrogen Status"].dropna().astype(str).unique().tolist()),
            "Progesterone Status": sorted(self.df["Progesterone Status"].dropna().astype(str).unique().tolist()),
        }

    @staticmethod
    def _load_data() -> pd.DataFrame:
        df = pd.read_csv(DATA_PATH)
        df.columns = df.columns.str.strip()
        df["event"] = (df["Status"].astype(str).str.strip() == "Dead").astype(int)
        df["duration"] = pd.to_numeric(df["Survival Months"], errors="coerce")

        for col in RAW_FEATURES:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        return df

    @staticmethod
    def _build_train_test(df: pd.DataFrame):
        model_df = df[RAW_FEATURES + ["duration", "event"]].dropna().copy()

        X = pd.get_dummies(model_df[RAW_FEATURES], drop_first=True).astype(float)
        y = Surv.from_arrays(
            event=model_df["event"].astype(bool).values,
            time=model_df["duration"].astype(float).values,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=model_df["event"],
        )
        return X_train, X_test, y_train, y_test

    def _prepare_one_patient(
        self,
        age,
        tumor_size,
        reginol_node_positive,
        n_stage,
        grade,
        estrogen_status,
        progesterone_status,
        t_stage,
        regional_node_examined,
    ):
        patient = {
            "Age": float(age),
            "Tumor Size": float(tumor_size),
            "Regional Node Examined": float(regional_node_examined),
            "Reginol Node Positive": float(reginol_node_positive),
            "T Stage": str(t_stage),
            "N Stage": str(n_stage),
            "Grade": str(grade),
            "A Stage": str(self.defaults_num["A Stage"]),
            "Estrogen Status": str(estrogen_status),
            "Progesterone Status": str(progesterone_status),
        }

        patient_df = pd.DataFrame([patient])
        patient_X = pd.get_dummies(patient_df, drop_first=True)
        patient_X = patient_X.reindex(columns=self.feature_columns, fill_value=0).astype(float)
        return patient_X

    def predict(
        self,
        age,
        tumor_size,
        reginol_node_positive,
        n_stage,
        grade,
        estrogen_status,
        progesterone_status,
        t_stage,
        regional_node_examined,
    ):
        patient_X = self._prepare_one_patient(
            age=age,
            tumor_size=tumor_size,
            reginol_node_positive=reginol_node_positive,
            n_stage=n_stage,
            grade=grade,
            estrogen_status=estrogen_status,
            progesterone_status=progesterone_status,
            t_stage=t_stage,
            regional_node_examined=regional_node_examined,
        )

        surv_fn = self.rsf.predict_survival_function(patient_X)[0]
        domain_min, domain_max = surv_fn.domain
        domain_max_int = int(np.floor(domain_max))

        months = np.arange(int(domain_min), domain_max_int + 1)
        surv_prob = np.array([surv_fn(m) for m in months])

        # Si 60 mois sort du domaine appris, on utilise la borne max disponible.
        t_5y_eval = min(60, domain_max_int)
        risk_5y = float(1.0 - surv_fn(t_5y_eval))

        fig_surv = go.Figure()
        fig_surv.add_trace(
            go.Scatter(
                x=months,
                y=surv_prob,
                mode="lines",
                name="Survival probability",
                line={"color": "#0F766E", "width": 3},
                hovertemplate="Month: %{x}<br>Survival probability: %{y:.3f}<extra></extra>",
            )
        )
        fig_surv.update_layout(
            title="Courbe de survie personnalisee (RSF)",
            xaxis_title="Temps (mois)",
            yaxis_title="Probabilite de survie",
            yaxis={"range": [0, 1]},
            template="simple_white",
            hovermode="x",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )

        shap_values = self.shap_explainer(patient_X)

        fig_shap = plt.figure(figsize=(10, 5.5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        plt.tight_layout()

        risk_label = "Risque estime de deces avant 60 mois" if t_5y_eval == 60 else f"Risque estime de deces avant {t_5y_eval} mois"

        risk_md = (
            "### Score de risque a 5 ans\n"
            f"**{risk_label} : {risk_5y * 100:.1f}%**\n\n"
            f"Probabilite de survie a 5 ans : **{(1 - risk_5y) * 100:.1f}%**"
        )

        return fig_surv, risk_md, fig_shap


def build_interface(service: RSFSurvivalService):
    css = """
    .gradio-container {max-width: 1100px !important; margin: auto;}
    .title {font-size: 28px; font-weight: 700; margin-bottom: 6px;}
    .subtitle {opacity: 0.85; margin-bottom: 14px;}
    """

    with Blocks(theme=gr.themes.Soft(), css=css, title="Onco Survival RSF") as demo:
        Markdown("<div class='title'>Onco Survival Explorer</div>")
        Markdown(
            "<div class='subtitle'>"
            "Predire la survie personnalisee, estimer le risque a 5 ans et expliquer la prediction avec SHAP."
            "</div>"
        )

        with Row():
            with Column(scale=1):
                age = Slider(20, 95, value=60, step=1, label="Age")
                tumor_size = Number(value=30, label="Taille tumeur")
                reginol_node_positive = Number(
                    value=service.defaults_num["Reginol Node Positive"],
                    label="Reginol Node Positive",
                )
                n_stage = Dropdown(service.choices["N Stage"], value=service.defaults_num["N Stage"], label="N Stage")
                grade = Dropdown(service.choices["Grade"], value=service.choices["Grade"][0], label="Grade")
                estrogen = Dropdown(
                    service.choices["Estrogen Status"],
                    value=service.choices["Estrogen Status"][0],
                    label="Estrogen Status",
                )
                progesterone = Dropdown(
                    service.choices["Progesterone Status"],
                    value=service.choices["Progesterone Status"][0],
                    label="Progesterone Status",
                )
                t_stage = Dropdown(service.choices["T Stage"], value=service.choices["T Stage"][0], label="T Stage")
                regional_node_examined = Number(
                    value=service.defaults_num["Regional Node Examined"],
                    label="Regional Node Examined",
                )
                run_btn = gr.Button("Predire", variant="primary")

            with Column(scale=2):
                out_curve = Plot(label="Output 1 - Courbe de survie personnalisee (interactive)")
                out_risk = Markdown(label="Output 2 - Score de risque a 5 ans")
                out_shap = Plot(label="Output 3 - SHAP explicatif")

        run_btn.click(
            fn=service.predict,
            inputs=[
                age,
                tumor_size,
                reginol_node_positive,
                n_stage,
                grade,
                estrogen,
                progesterone,
                t_stage,
                regional_node_examined,
            ],
            outputs=[out_curve, out_risk, out_shap],
        )

    return demo


def main():
    service = RSFSurvivalService()
    app = build_interface(service)
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port_env = os.getenv("GRADIO_SERVER_PORT")
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    if port_env:
        app.launch(server_name=host, server_port=int(port_env), share=share)
        return

    # Default behavior: try common local ports before letting Gradio choose one.
    for port in [7860, 7861, 7862, 7863, 7864, 7865]:
        try:
            app.launch(server_name=host, server_port=port, share=share)
            return
        except OSError as exc:
            if "Cannot find empty port" not in str(exc):
                raise

    app.launch(server_name=host, share=share)


if __name__ == "__main__":
    main()
