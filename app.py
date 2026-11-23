
# ============================================================
# SENTINELXAI
# Explainable AI for Intelligent Cyber Risk Assessment
# COMPLETE BACKEND + PROFESSIONAL PDF REPORT V3
# ============================================================

import os
import uuid
from io import BytesIO
from datetime import datetime, timezone
from xml.sax.saxutils import escape

import joblib
import numpy as np
import pandas as pd
import shap

from flask import (
    Flask, render_template, request,
    redirect, url_for, session,
    send_file, abort
)

from werkzeug.utils import secure_filename

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, KeepTogether,
    HRFlowable
)
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line
)


# ============================================================
# 1. FLASK CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SENTINELXAI_SECRET_KEY",
    "sentinelxai_local_development_key"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

RESULT_COLUMNS = [
    "RF_Probability",
    "XGB_Probability",
    "LR_Probability",
    "Ensemble_Risk",
    "Prediction",
    "Severity"
]

EXCLUDE_COLUMNS = [
    "id", "label", "attack_cat"
] + RESULT_COLUMNS


# ============================================================
# 2. LOAD TRAINED MODELS
# ============================================================

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing model file: {filename}"
        )

    return joblib.load(path)


print("Loading SentinelXAI ML models...")

rf_model = load_model("sentinelxai_rf.pkl")
xgb_model = load_model("sentinelxai_xgb.pkl")
lr_model = load_model("sentinelxai_lr.pkl")
feature_names = list(
    load_model("sentinelxai_features.pkl")
)
scaler = load_model("sentinelxai_scaler.pkl")

print("All SentinelXAI models loaded successfully.")


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() == "csv"
    )


def format_feature_value(value):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass

    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return f"{value:.6g}"

    return str(value)


def get_risk_level(score):
    if score < 30:
        return "Low"
    if score < 60:
        return "Medium"
    if score < 80:
        return "High"
    return "Critical"


def get_recommendation(severity):
    recommendations = {
        "Low": (
            "Continue Monitoring",
            "Maintain routine monitoring and review "
            "related activity if new indicators appear."
        ),
        "Medium": (
            "Review Activity",
            "Inspect related network flows and check "
            "for repeated or unusual behaviour."
        ),
        "High": (
            "Priority Investigation",
            "Investigate this event, correlate related "
            "logs and verify potentially affected assets."
        ),
        "Critical": (
            "Immediate Investigation",
            "Prioritize analyst triage, preserve relevant "
            "evidence and validate the threat before "
            "considering containment actions."
        )
    }

    return recommendations.get(
        severity,
        ("Security Review", "Review this network event.")
    )


def get_result_path():
    path = session.get("result_path")

    if path and os.path.isfile(path):
        return path

    return None


# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

def preprocess_data(df):
    data = df.copy()

    data = data.drop(
        columns=EXCLUDE_COLUMNS,
        errors="ignore"
    )

    categorical_columns = [
        column
        for column in ["proto", "service", "state"]
        if column in data.columns
    ]

    if categorical_columns:
        data = pd.get_dummies(
            data,
            columns=categorical_columns
        )

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    data = data.reindex(
        columns=feature_names,
        fill_value=0
    )

    return data.astype(float)


# ============================================================
# 5. ML ENSEMBLE PREDICTION
# ============================================================

def ensemble_predict(X):
    rf_prob = rf_model.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]

    X_scaled = scaler.transform(X)
    lr_prob = lr_model.predict_proba(X_scaled)[:, 1]

    ensemble_prob = (
        0.40 * rf_prob
        + 0.40 * xgb_prob
        + 0.20 * lr_prob
    )

    prediction = (
        ensemble_prob >= 0.50
    ).astype(int)

    return (
        rf_prob,
        xgb_prob,
        lr_prob,
        ensemble_prob,
        prediction
    )


# ============================================================
# 6. SHAP - RANDOM FOREST ONLY
# ============================================================

def get_shap_explanation(X_record):
    try:
        explainer = shap.TreeExplainer(rf_model)

        result = explainer.shap_values(
            X_record,
            check_additivity=False
        )

        if isinstance(result, list):
            values = np.asarray(
                result[1]
                if len(result) > 1
                else result[0]
            )
        else:
            values = np.asarray(result)

        if values.ndim == 3:
            values = values[0, :, 1]

        elif values.ndim == 2:
            values = values[0]

        values = np.asarray(values).flatten()

        if len(values) != len(X_record.columns):
            return []

        explanation = [
            {
                "feature": str(feature),
                "value": float(value)
            }
            for feature, value in zip(
                X_record.columns,
                values
            )
        ]

        explanation.sort(
            key=lambda item: abs(item["value"]),
            reverse=True
        )

        return explanation[:10]

    except Exception as exc:
        app.logger.warning(
            "SHAP explanation error: %s",
            exc
        )
        return []


# ============================================================
# 7. NETWORK FORENSICS
# ============================================================

NETWORK_FIELDS = [
    ("Duration", "dur"),
    ("Source Packets", "spkts"),
    ("Destination Packets", "dpkts"),
    ("Source Bytes", "sbytes"),
    ("Destination Bytes", "dbytes"),
    ("Source TTL", "sttl"),
    ("Destination TTL", "dttl"),
    ("Source Loss", "sloss"),
    ("Destination Loss", "dloss"),
    ("Source Mean", "smean"),
    ("Destination Mean", "dmean"),
    ("Source Load", "sload"),
    ("Destination Load", "dload"),
    ("Source Jitter", "sjit"),
    ("Destination Jitter", "djit"),
    ("Source TCP RTT", "tcprtt"),
    ("SYN-ACK", "synack"),
    ("ACK-DAT", "ackdat"),
    ("Source Interpacket", "sinpkt"),
    ("Destination Interpacket", "dinpkt")
]


def build_network_inspection(row):
    return [
        {
            "name": name,
            "field": field,
            "value": format_feature_value(row[field])
        }
        for name, field in NETWORK_FIELDS
        if field in row.index
    ]


def build_feature_metrics(row):
    metrics = []

    for _, field in NETWORK_FIELDS:
        if field not in row.index:
            continue

        try:
            value = float(row[field])

            if np.isfinite(value):
                metrics.append({
                    "feature": field,
                    "value": value
                })

        except (TypeError, ValueError):
            continue

    return metrics


def build_protocol_info(row):
    return {
        "protocol": format_feature_value(
            row.get("proto", "N/A")
        ),
        "service": format_feature_value(
            row.get("service", "N/A")
        ),
        "state": format_feature_value(
            row.get("state", "N/A")
        )
    }


# ============================================================
# 8. DATASET LABEL
# ============================================================

def get_dataset_label(row):
    value = row.get("label", None)

    if value is None:
        return "NOT AVAILABLE"

    try:
        if pd.isna(value):
            return "NOT AVAILABLE"

        return (
            "ATTACK"
            if int(value) == 1
            else "NORMAL"
        )

    except (TypeError, ValueError):
        return str(value)


# ============================================================
# 9. LOAD EVENT
# ============================================================

def load_event(record_number):
    path = get_result_path()

    if not path or record_number < 1:
        return None

    df = pd.read_csv(
        path,
        skiprows=lambda line: (
            line > 0 and line != record_number
        ),
        nrows=1
    )

    if df.empty:
        return None

    return df.iloc[0]


# ============================================================
# 10. EVENT ASSESSMENT
# ============================================================

def assess_event(
    row,
    record_number,
    include_shap=True
):
    original = row.drop(
        labels=RESULT_COLUMNS,
        errors="ignore"
    )

    X_record = preprocess_data(
        pd.DataFrame([original])
    )

    saved_results = all(
        column in row.index
        for column in RESULT_COLUMNS
    )

    if saved_results:
        rf_score = float(row["RF_Probability"])
        xgb_score = float(row["XGB_Probability"])
        lr_score = float(row["LR_Probability"])
        risk_score = float(row["Ensemble_Risk"])

        status = str(row["Prediction"]).upper()
        severity = str(row["Severity"])

    else:
        rf, xgb, lr, final, pred = (
            ensemble_predict(X_record)
        )

        rf_score = float(rf[0] * 100)
        xgb_score = float(xgb[0] * 100)
        lr_score = float(lr[0] * 100)
        risk_score = float(final[0] * 100)

        status = (
            "ATTACK"
            if int(pred[0]) == 1
            else "NORMAL"
        )

        severity = get_risk_level(risk_score)

    dataset_label = get_dataset_label(row)

    if dataset_label == "NOT AVAILABLE":
        comparison = "REFERENCE LABEL NOT AVAILABLE"
    elif dataset_label == status:
        comparison = "AGREEMENT"
    else:
        comparison = "DISAGREEMENT"

    recommendation_title, recommendation_text = (
        get_recommendation(severity)
    )

    profiles = {
        "Low": "Low-Risk Network Activity",
        "Medium": "Moderate-Risk Network Activity",
        "High": "High-Risk Network Activity",
        "Critical": "Critical-Risk Network Activity"
    }

    risk = {
        "risk_score": risk_score,
        "severity": severity,
        "status": status,
        "prediction": (
            1 if status == "ATTACK" else 0
        ),
        "rf_probability": rf_score,
        "xgb_probability": xgb_score,
        "lr_probability": lr_score
    }

    raw_event_data = [
        {
            "field": str(column),
            "value": format_feature_value(
                row[column]
            )
        }
        for column in row.index
        if column not in RESULT_COLUMNS
    ]

    return {
        "record_number": record_number,
        "risk": risk,
        "dataset_label": dataset_label,
        "comparison": comparison,
        "threat_profile": profiles.get(
            severity,
            "Network Activity"
        ),
        "recommendation_title": recommendation_title,
        "recommendation_text": recommendation_text,
        "analyst_decision": session.get(
            f"decision_{record_number}",
            "Not yet selected"
        ),
        "shap_explanation": (
            get_shap_explanation(X_record)
            if include_shap else []
        ),
        "network_inspection": build_network_inspection(
            row
        ),
        "feature_metrics": build_feature_metrics(
            row
        ),
        "protocol_info": build_protocol_info(row),
        "raw_event_data": raw_event_data
    }


# ============================================================
# 11. HOME AND ANALYSIS
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    return render_template("analysis.html")


# ============================================================
# 12. CSV UPLOAD
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or not file.filename:
        return "Please select a CSV file.", 400

    if not allowed_file(file.filename):
        return "Only CSV files are supported.", 400

    filename = (
        str(uuid.uuid4())
        + "_"
        + secure_filename(file.filename)
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)

    session["dataset_path"] = filepath

    return redirect(url_for("run_analysis"))


# ============================================================
# 13. RUN ANALYSIS
# ============================================================

@app.route("/run-analysis")
def run_analysis():
    dataset_path = session.get("dataset_path")

    if (
        not dataset_path
        or not os.path.isfile(dataset_path)
    ):
        return redirect(url_for("analyze"))

    try:
        df = pd.read_csv(dataset_path)

        if df.empty:
            return "Uploaded CSV is empty.", 400

        X = preprocess_data(df)

        rf, xgb, lr, final, prediction = (
            ensemble_predict(X)
        )

        result_df = df.copy()

        result_df["RF_Probability"] = rf * 100
        result_df["XGB_Probability"] = xgb * 100
        result_df["LR_Probability"] = lr * 100
        result_df["Ensemble_Risk"] = final * 100

        result_df["Prediction"] = np.where(
            prediction == 1,
            "ATTACK",
            "NORMAL"
        )

        result_df["Severity"] = [
            get_risk_level(float(value * 100))
            for value in final
        ]

        result_path = os.path.join(
            UPLOAD_FOLDER,
            "latest_analysis.csv"
        )

        result_df.to_csv(
            result_path,
            index=False
        )

        session["result_path"] = result_path

        print(
            "Analysis completed:",
            len(result_df),
            "events"
        )

        return redirect(url_for("monitoring"))

    except Exception as exc:
        app.logger.exception("CSV analysis failed")

        return (
            "Analysis error: "
            + escape(str(exc)),
            500
        )


# ============================================================
# 14. SOC DASHBOARD
# ============================================================

@app.route("/monitoring")
def monitoring():
    path = get_result_path()

    if not path:
        return redirect(url_for("analyze"))

    df = pd.read_csv(
        path,
        usecols=[
            "Prediction",
            "Severity",
            "Ensemble_Risk"
        ]
    )

    total_records = len(df)

    attack_count = int(
        (df["Prediction"] == "ATTACK").sum()
    )

    normal_count = int(
        (df["Prediction"] == "NORMAL").sum()
    )

    severity_counts = (
        df["Severity"].value_counts()
    )

    low_count = int(
        severity_counts.get("Low", 0)
    )
    medium_count = int(
        severity_counts.get("Medium", 0)
    )
    high_count = int(
        severity_counts.get("High", 0)
    )
    critical_count = int(
        severity_counts.get("Critical", 0)
    )

    average_risk = (
        float(df["Ensemble_Risk"].mean())
        if total_records else 0.0
    )

    events = []

    for index, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):
        risk_value = round(
            float(row.Ensemble_Risk),
            2
        )

        events.append({
            "record": index,
            "record_number": index,
            "id": index,
            "status": str(row.Prediction),
            "prediction": str(row.Prediction),
            "risk": risk_value,
            "risk_score": risk_value,
            "severity": str(row.Severity)
        })

    return render_template(
        "monitoring.html",
        total_records=total_records,
        total_events=total_records,
        attack_count=attack_count,
        detected_attacks=attack_count,
        normal_count=normal_count,
        normal_traffic=normal_count,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        critical_count=critical_count,
        high_critical_count=(
            high_count + critical_count
        ),
        average_risk=round(average_risk, 2),
        events=events
    )


# ============================================================
# 15. EVENT INVESTIGATION
# ============================================================

@app.route("/record/<int:record_number>")
def record_details(record_number):
    if not get_result_path():
        return redirect(url_for("analyze"))

    row = load_event(record_number)

    if row is None:
        abort(404)

    context = assess_event(
        row,
        record_number,
        include_shap=True
    )

    return render_template(
        "record.html",
        **context
    )


# ============================================================
# 16. ANALYST DECISION
# ============================================================

@app.route(
    "/record/<int:record_number>/decision",
    methods=["POST"]
)
def analyst_decision(record_number):
    allowed = {
        "Confirmed Attack",
        "Marked as Normal",
        "Needs Review"
    }

    decision = request.form.get("decision")

    if decision not in allowed:
        return "Invalid analyst decision.", 400

    if load_event(record_number) is None:
        abort(404)

    session[
        f"decision_{record_number}"
    ] = decision

    return redirect(
        url_for(
            "record_details",
            record_number=record_number,
            tab="decision"
        )
    )


# ============================================================
# 17. OTHER ROUTES
# ============================================================

@app.route("/processing")
def processing():
    return render_template("processing.html")


@app.route("/result")
def result():
    return redirect(url_for("monitoring"))


# ============================================================
# 18. PROFESSIONAL PDF REPORT V3
# ============================================================

@app.route("/record/<int:record_number>/report")
def download_security_report(record_number):
    if not get_result_path():
        return redirect(url_for("analyze"))

    row = load_event(record_number)

    if row is None:
        abort(404)

    try:
        data = assess_event(
            row,
            record_number,
            include_shap=True
        )

        risk = data["risk"]

        # ------------------------------------------
        # COLORS
        # ------------------------------------------

        NAVY = colors.HexColor("#0A1B2D")
        CYAN = colors.HexColor("#00BFEA")
        LIGHT = colors.HexColor("#EDF7FC")
        MUTED = colors.HexColor("#526579")
        GREEN = colors.HexColor("#139E85")
        RED = colors.HexColor("#D95364")
        AMBER = colors.HexColor("#D38B27")
        ORANGE = colors.HexColor("#E77C38")
        WHITE = colors.white

        severity_colors = {
            "Low": GREEN,
            "Medium": AMBER,
            "High": ORANGE,
            "Critical": RED
        }

        severity_color = severity_colors.get(
            risk["severity"],
            CYAN
        )

        content_width = A4[0] - 84

        # ------------------------------------------
        # DOCUMENT
        # ------------------------------------------

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=42,
            rightMargin=42,
            topMargin=55,
            bottomMargin=50,
            title=(
                "SentinelXAI Security Analysis "
                f"Event {record_number}"
            ),
            author="SentinelXAI Security Operations"
        )

        # ------------------------------------------
        # STYLES
        # ------------------------------------------

        title_style = ParagraphStyle(
            "V3Title",
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            spaceAfter=5
        )

        subtitle_style = ParagraphStyle(
            "V3Subtitle",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=8
        )

        section_style = ParagraphStyle(
            "V3Section",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=7
        )

        body_style = ParagraphStyle(
            "V3Body",
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=NAVY,
            spaceAfter=6
        )

        small_style = ParagraphStyle(
            "V3Small",
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
            spaceAfter=5
        )

        cell_style = ParagraphStyle(
            "V3Cell",
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=NAVY
        )

        header_style = ParagraphStyle(
            "V3Header",
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=WHITE
        )

        # ------------------------------------------
        # HELPERS
        # ------------------------------------------

        def p(value, style=None):
            return Paragraph(
                escape(str(value)),
                style or cell_style
            )

        def heading(number, label):
            return Paragraph(
                f"{number} | {escape(label)}",
                section_style
            )

        def make_table(headers, rows, widths=None):
            table_data = [
                [
                    p(header, header_style)
                    for header in headers
                ]
            ]

            for values in rows:
                table_data.append([
                    p(value)
                    for value in values
                ])

            table = Table(
                table_data,
                colWidths=widths,
                repeatRows=1,
                hAlign="LEFT"
            )

            table.setStyle(TableStyle([
                (
                    "BACKGROUND",
                    (0, 0), (-1, 0),
                    NAVY
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1), (-1, -1),
                    [WHITE, LIGHT]
                ),
                (
                    "VALIGN",
                    (0, 0), (-1, -1),
                    "TOP"
                ),
                (
                    "LEFTPADDING",
                    (0, 0), (-1, -1),
                    9
                ),
                (
                    "RIGHTPADDING",
                    (0, 0), (-1, -1),
                    9
                ),
                (
                    "TOPPADDING",
                    (0, 0), (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0), (-1, -1),
                    6
                ),
                (
                    "LINEBELOW",
                    (0, -1), (-1, -1),
                    0.6, CYAN
                )
            ]))

            return table

        # ------------------------------------------
        # RISK SCORE CHART
        # ------------------------------------------

        def risk_chart(score):
            width = content_width
            height = 72

            drawing = Drawing(width, height)

            drawing.add(Rect(
                0, 0, width, height,
                rx=8, ry=8,
                fillColor=NAVY,
                strokeColor=None
            ))

            drawing.add(String(
                16, 48,
                "WEIGHTED ENSEMBLE RISK SCORE",
                fontName="Helvetica-Bold",
                fontSize=9,
                fillColor=WHITE
            ))

            drawing.add(String(
                width - 16, 46,
                f"{score:.2f}%",
                fontName="Helvetica-Bold",
                fontSize=18,
                fillColor=WHITE,
                textAnchor="end"
            ))

            drawing.add(Rect(
                16, 17,
                width - 32, 11,
                fillColor=colors.HexColor("#244052"),
                strokeColor=None
            ))

            drawing.add(Rect(
                16, 17,
                (width - 32) * (
                    max(0, min(score, 100)) / 100
                ),
                11,
                fillColor=severity_color,
                strokeColor=None
            ))

            return drawing

        # ------------------------------------------
        # ML COMPARISON CHART
        # ------------------------------------------

        def ml_chart():
            models = [
                (
                    "Random Forest",
                    risk["rf_probability"],
                    CYAN
                ),
                (
                    "XGBoost",
                    risk["xgb_probability"],
                    GREEN
                ),
                (
                    "Logistic Regression",
                    risk["lr_probability"],
                    AMBER
                ),
                (
                    "Weighted Ensemble",
                    risk["risk_score"],
                    severity_color
                )
            ]

            width = content_width
            height = 120
            drawing = Drawing(width, height)

            bar_x = 145
            bar_width = width - 215

            for index, (name, value, color) in enumerate(models):
                y = height - 22 - index * 29

                drawing.add(String(
                    0, y, name,
                    fontName="Helvetica",
                    fontSize=9,
                    fillColor=NAVY
                ))

                drawing.add(Rect(
                    bar_x, y - 3,
                    bar_width, 11,
                    fillColor=LIGHT,
                    strokeColor=None
                ))

                drawing.add(Rect(
                    bar_x, y - 3,
                    bar_width * (
                        max(0, min(value, 100)) / 100
                    ),
                    11,
                    fillColor=color,
                    strokeColor=None
                ))

                drawing.add(String(
                    width - 2, y,
                    f"{value:.2f}%",
                    fontName="Helvetica-Bold",
                    fontSize=9,
                    fillColor=NAVY,
                    textAnchor="end"
                ))

            return drawing

        # ------------------------------------------
        # SHAP CONTRIBUTION CHART
        # ------------------------------------------

        def shap_chart(items):
            top_items = items[:7]

            width = content_width
            height = 28 * len(top_items) + 22

            drawing = Drawing(width, height)

            middle = 282
            max_bar_width = 145

            max_value = max(
                abs(float(item["value"]))
                for item in top_items
            ) or 1.0

            drawing.add(Line(
                middle, 8,
                middle, height - 6,
                strokeColor=MUTED,
                strokeWidth=0.7
            ))

            for index, item in enumerate(top_items):
                value = float(item["value"])
                y = height - 22 - index * 28

                drawing.add(String(
                    0, y,
                    str(item["feature"])[:28],
                    fontName="Helvetica",
                    fontSize=8.5,
                    fillColor=NAVY
                ))

                bar_width = (
                    max_bar_width
                    * abs(value)
                    / max_value
                )

                if bar_width > 0:
                    drawing.add(Rect(
                        middle if value >= 0
                        else middle - bar_width,
                        y - 3,
                        bar_width,
                        11,
                        fillColor=(
                            RED if value >= 0
                            else GREEN
                        ),
                        strokeColor=None
                    ))

                drawing.add(String(
                    width - 2, y,
                    f"{value:+.5f}",
                    fontName="Helvetica",
                    fontSize=8,
                    fillColor=NAVY,
                    textAnchor="end"
                ))

            return drawing

        # ------------------------------------------
        # REPORT METADATA
        # ------------------------------------------

        generated = datetime.now(timezone.utc)

        report_id = (
            f"SX-{generated.strftime('%Y%m%d')}-"
            f"{record_number:06d}"
        )

        story = []

        story.append(
            Paragraph("SentinelXAI", title_style)
        )

        story.append(
            Paragraph(
                "PROFESSIONAL SECURITY ANALYSIS REPORT",
                subtitle_style
            )
        )

        story.append(HRFlowable(
            width="100%",
            thickness=2,
            color=CYAN,
            spaceAfter=10
        ))

        story.append(make_table(
            ["REPORT ID", "EVENT", "GENERATED (UTC)"],
            [
                (
                    report_id,
                    f"#{record_number}",
                    generated.strftime(
                        "%d %b %Y %H:%M"
                    )
                )
            ],
            [
                205,
                90,
                content_width - 295
            ]
        ))

        # ------------------------------------------
        # 01 - EXECUTIVE SUMMARY
        # ------------------------------------------

        story.append(heading(
            "01", "Executive Summary"
        ))

        story.append(risk_chart(
            float(risk["risk_score"])
        ))

        story.append(Spacer(1, 9))

        story.append(make_table(
            ["Assessment Field", "Result"],
            [
                ("AI Prediction", risk["status"]),
                ("Severity", risk["severity"]),
                ("Dataset Label", data["dataset_label"]),
                ("AI vs Dataset", data["comparison"]),
                (
                    "Analyst Decision",
                    data["analyst_decision"]
                )
            ],
            [
                215,
                content_width - 215
            ]
        ))

        if data["comparison"] == "DISAGREEMENT":
            story.append(Spacer(1, 7))
            story.append(p(
                "Review flag: AI classification differs "
                "from the dataset reference label. "
                "Manual verification is recommended.",
                body_style
            ))

        # ------------------------------------------
        # 02 - ML COMPARISON
        # ------------------------------------------

        story.append(heading(
            "02", "Machine Learning Comparison"
        ))

        story.append(ml_chart())

        story.append(Spacer(1, 5))

        story.append(make_table(
            [
                "ML Model",
                "Weight",
                "Attack Probability"
            ],
            [
                (
                    "Random Forest",
                    "40%",
                    f'{risk["rf_probability"]:.2f}%'
                ),
                (
                    "XGBoost",
                    "40%",
                    f'{risk["xgb_probability"]:.2f}%'
                ),
                (
                    "Logistic Regression",
                    "20%",
                    f'{risk["lr_probability"]:.2f}%'
                ),
                (
                    "Weighted Ensemble",
                    "100%",
                    f'{risk["risk_score"]:.2f}%'
                )
            ],
            [
                210,
                100,
                content_width - 310
            ]
        ))

        # V3 FIX:
        # No trailing ML paragraph after the table.
        # Prevents an almost-empty overflow page.

        # ------------------------------------------
        # 03 - SHAP
        # ------------------------------------------

        shap_values = data["shap_explanation"]

        # V3 FIX:
        # No forced PageBreak.
        # Keep SHAP heading and explanation together.

        shap_intro = [
            heading(
                "03",
                "Explainable AI - SHAP"
            ),
            p(
                "SHAP values explain the Random Forest "
                "component only, not the full weighted "
                "ensemble. Positive contributions increase "
                "the explained attack-class output relative "
                "to its baseline; negative contributions "
                "decrease it. They do not prove causation.",
                body_style
            )
        ]

        if shap_values:
            shap_intro.append(
                shap_chart(shap_values)
            )

        story.append(KeepTogether(shap_intro))

        if shap_values:
            story.append(Spacer(1, 9))

            shap_rows = []

            for item in shap_values:
                value = float(item["value"])

                shap_rows.append((
                    item["feature"],
                    f"{value:+.6f}",
                    (
                        "Positive" if value >= 0
                        else "Negative"
                    )
                ))

            story.append(make_table(
                [
                    "Feature",
                    "SHAP Value",
                    "Direction"
                ],
                shap_rows,
                [
                    205,
                    155,
                    content_width - 360
                ]
            ))

        else:
            story.append(p(
                "SHAP explanation is unavailable "
                "for this event.",
                body_style
            ))

        # ------------------------------------------
        # 04 - NETWORK FORENSICS
        # ------------------------------------------

        protocol = data["protocol_info"]

        network_rows = [
            ("Protocol", protocol["protocol"]),
            ("Service", protocol["service"]),
            ("Connection State", protocol["state"])
        ]

        for item in data["network_inspection"]:
            network_rows.append((
                item["name"],
                item["value"]
            ))

        # V3 FIX:
        # Keep section heading with its table header
        # and first few rows, not the entire long table.

        story.append(KeepTogether([
            heading(
                "04",
                "Network Traffic Forensics"
            ),
            make_table(
                [
                    "Network Attribute",
                    "Observed Value"
                ],
                network_rows[:4],
                [
                    255,
                    content_width - 255
                ]
            )
        ]))

        if len(network_rows) > 4:
            story.append(make_table(
                [
                    "Network Attribute",
                    "Observed Value"
                ],
                network_rows[4:],
                [
                    255,
                    content_width - 255
                ]
            ))

        # ------------------------------------------
        # 05 - SECURITY RECOMMENDATIONS
        # ------------------------------------------

        recommendations = []

        if risk["severity"] in ("High", "Critical"):
            recommendations.append(
                "Prioritize analyst triage and correlate "
                "the event with related security logs."
            )

        elif risk["severity"] == "Medium":
            recommendations.append(
                "Review associated network flows for "
                "repeated or unusual patterns."
            )

        else:
            recommendations.append(
                "Maintain routine monitoring and reassess "
                "if correlated indicators appear."
            )

        if data["comparison"] == "DISAGREEMENT":
            recommendations.append(
                "Investigate the disagreement between "
                "the model prediction and reference label."
            )

        if risk["status"] == "ATTACK":
            recommendations.append(
                "Validate threat indicators and affected "
                "assets before considering containment."
            )

        if shap_values:
            leaders = ", ".join(
                str(item["feature"])
                for item in shap_values[:3]
            )

            recommendations.append(
                "Review the leading Random Forest SHAP "
                f"features: {leaders}."
            )

        if data["analyst_decision"] == "Needs Review":
            recommendations.append(
                "Keep this event under analyst review "
                "until the investigation is completed."
            )

        recommendations.append(
            "Preserve the original CSV event record "
            "and relevant investigation evidence."
        )

        # V3 FIX:
        # Heading + introduction + first recommendation
        # stay together. Remaining items flow naturally.

        recommendation_intro = [
            heading(
                "05",
                "Security Recommendations"
            ),
            p(
                data["recommendation_title"]
                + ": "
                + data["recommendation_text"],
                body_style
            )
        ]

        if recommendations:
            recommendation_intro.append(
                p(
                    f"1. {recommendations[0]}",
                    body_style
                )
            )

        story.append(KeepTogether(
            recommendation_intro
        ))

        for index, recommendation in enumerate(
            recommendations[1:],
            start=2
        ):
            story.append(p(
                f"{index}. {recommendation}",
                body_style
            ))

        # ------------------------------------------
        # 06 - ANALYST REVIEW
        # ------------------------------------------

        story.append(KeepTogether([
            heading(
                "06",
                "Analyst Review"
            ),
            make_table(
                [
                    "Review Field",
                    "Recorded Value"
                ],
                [
                    (
                        "Analyst Decision",
                        data["analyst_decision"]
                    ),
                    (
                        "AI Prediction",
                        risk["status"]
                    ),
                    (
                        "Dataset Label",
                        data["dataset_label"]
                    ),
                    (
                        "Comparison",
                        data["comparison"]
                    )
                ],
                [
                    215,
                    content_width - 215
                ]
            )
        ]))

        # ------------------------------------------
        # 07 - METHODOLOGY
        # ------------------------------------------

        story.append(KeepTogether([
            heading(
                "07",
                "Methodology and Limitations"
            ),
            p(
                "This report is generated from an uploaded "
                "and processed CSV dataset. It represents "
                "offline batch analysis, not live network "
                "monitoring. The weighted ensemble uses "
                "Random Forest (40%), XGBoost (40%) and "
                "Logistic Regression (20%). Model outputs "
                "require analyst validation. SHAP explains "
                "the Random Forest component only. Analyst "
                "decisions are stored in the current Flask "
                "session, not a persistent audit trail.",
                body_style
            )
        ]))

        # ------------------------------------------
        # PROFESSIONAL PAGE HEADER / FOOTER
        # ------------------------------------------

        def draw_page(canvas, pdf_doc):
            canvas.saveState()

            page_width, page_height = A4

            canvas.setFillColor(NAVY)

            canvas.rect(
                0,
                page_height - 31,
                page_width,
                31,
                fill=1,
                stroke=0
            )

            canvas.setFont(
                "Helvetica-Bold",
                9
            )

            canvas.setFillColor(WHITE)

            canvas.drawString(
                42,
                page_height - 20,
                "SENTINELXAI / SECURITY OPERATIONS"
            )

            canvas.setStrokeColor(CYAN)
            canvas.setLineWidth(0.8)

            canvas.line(
                42,
                35,
                page_width - 42,
                35
            )

            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)

            canvas.drawString(
                42,
                23,
                "CONFIDENTIAL - ANALYST REVIEW"
            )

            canvas.drawRightString(
                page_width - 42,
                23,
                f"{report_id} | PAGE {pdf_doc.page}"
            )

            canvas.restoreState()

        # ------------------------------------------
        # GENERATE PDF V3
        # ------------------------------------------

        doc.build(
            story,
            onFirstPage=draw_page,
            onLaterPages=draw_page
        )

        buffer.seek(0)

        filename = (
            "SentinelXAI_Security_Report_V3_"
            f"Event_{record_number}.pdf"
        )

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename
        )

    except Exception as exc:
        app.logger.exception(
            "PDF Report V3 generation failed"
        )

        return (
            "PDF generation error: "
            + escape(str(exc)),
            500
        )


# ============================================================
# 19. START APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
