import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Job Acceptance Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Job Acceptance Prediction System")


MODEL_PATH = "models/production_model.pkl"
DATA_PATH = "Data/cleaned_data.csv"
XCOL_PATH = "Data/X_ml_ready.csv"


if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH) and os.path.exists(XCOL_PATH)):
    st.error("❌ Required files not found. Run Streamlit from project root folder.")
    st.stop()


model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)
X_columns = pd.read_csv(XCOL_PATH).columns.tolist()


st.sidebar.header("🎛️ Filter Panel")

company_filter = st.sidebar.multiselect(
    "Company Tier",
    sorted(df["company_tier"].dropna().unique()),
    default=sorted(df["company_tier"].dropna().unique())
)

competition_filter = st.sidebar.multiselect(
    "Competition Level",
    sorted(df["competition_level"].dropna().unique()),
    default=sorted(df["competition_level"].dropna().unique())
)

experience_range = st.sidebar.slider(
    "Years of Experience",
    int(df["years_of_experience"].min()),
    int(df["years_of_experience"].max()),
    (0, int(df["years_of_experience"].max()))
)

filtered_df = df[
    (df["company_tier"].isin(company_filter)) &
    (df["competition_level"].isin(competition_filter)) &
    (df["years_of_experience"].between(experience_range[0], experience_range[1]))
].copy()

st.sidebar.success(f"🎯 Active Records: {len(filtered_df)}")
\
filtered_df["interview_score"] = (
    filtered_df["technical_score"] +
    filtered_df["aptitude_score"] +
    filtered_df["communication_score"]
) / 3

\
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Dashboard", "📈 Analytics", "🤖 Prediction", "⚠️ Risk", "📁 Explorer"]
)


with tab1:
    st.subheader("📊 Key Performance Indicators")

    total_candidates = len(filtered_df)
    placement_rate = (filtered_df["status"] == "Placed").mean() * 100
    avg_skills = filtered_df["skills_match_percentage"].mean()
    avg_interview = filtered_df["interview_score"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", total_candidates)
    col2.metric("Placement Rate (%)", f"{placement_rate:.1f}")
    col3.metric("Avg Skills Match (%)", f"{avg_skills:.1f}")
    col4.metric("Avg Interview Score", f"{avg_interview:.1f}")

    st.info("KPIs dynamically update based on sidebar filters.")


with tab2:
    st.subheader("📈 Visual Analytics")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("**Placement Distribution**")
        placement_counts = filtered_df["status"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.bar(placement_counts.index, placement_counts.values)
        ax1.set_xlabel("Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with colB:
        st.markdown("**Average Skills Match by Status**")
        skill_avg = filtered_df.groupby("status")["skills_match_percentage"].mean()
        fig2, ax2 = plt.subplots()
        ax2.bar(skill_avg.index, skill_avg.values)
        ax2.set_xlabel("Status")
        ax2.set_ylabel("Skills Match (%)")
        st.pyplot(fig2)


with tab3:
    st.subheader("🤖 Candidate Outcome Simulator")

    with st.form("prediction_form"):
        technical = st.slider("Technical Score", 0, 100, 70)
        aptitude = st.slider("Aptitude Score", 0, 100, 65)
        communication = st.slider("Communication Score", 0, 100, 70)
        skills_match = st.slider("Skills Match (%)", 0, 100, 75)
        experience = st.slider("Years of Experience", 0, 15, 2)
        company_tier = st.selectbox("Company Tier", sorted(df["company_tier"].unique()))
        competition = st.selectbox("Competition Level", sorted(df["competition_level"].unique()))
        role_match = st.selectbox("Job Role Match", sorted(df["job_role_match"].unique()))

        submit = st.form_submit_button("🔮 Predict")

    if submit:
        input_dict = {
            "technical_score": technical,
            "aptitude_score": aptitude,
            "communication_score": communication,
            "skills_match_percentage": skills_match,
            "years_of_experience": experience,
            "company_tier": company_tier,
            "competition_level": competition,
            "job_role_match": role_match
        }

        input_df = pd.DataFrame([input_dict])

        # One-hot encoding
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

        # Prediction
        prob = model.predict_proba(input_encoded)[0][1]
        prediction = model.predict(input_encoded)[0]

        st.metric("🎯 Placement Probability", f"{prob*100:.2f}%")

        if prediction == 1:
            st.success("✅ Candidate is likely to be PLACED")
        else:
            st.warning("⚠️ Candidate is likely to be NOT PLACED")


with tab4:
    st.subheader("⚠️ High Risk Candidates")

    df["interview_score"] = (
        df["technical_score"] +
        df["aptitude_score"] +
        df["communication_score"]
    ) / 3

    risk_df = df[
        (df["skills_match_percentage"] < 60) |
        (df["interview_score"] < 55) |
        (df["years_of_experience"] == 0)
    ]

    st.warning(f"High Risk Candidates Identified: {len(risk_df)}")
    st.dataframe(risk_df.head(40))


with tab5:
    st.subheader("📁 Dataset Explorer")

    selected_columns = st.multiselect(
        "Select Columns",
        filtered_df.columns.tolist(),
        default=filtered_df.columns[:6].tolist()
    )

    st.dataframe(filtered_df[selected_columns].head(50))
    st.info(f"Total records after filter: {len(filtered_df)}")
