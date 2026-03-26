import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tempfile
import os

st.set_page_config(layout="wide")
st.title("Global Methane Monitoring Dashboard")

# =====================================================
# Navigation State
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# =====================================================
# DATA UPLOAD SECTION (Always visible)
# =====================================================

st.sidebar.header("Upload Data")

sat_files = st.sidebar.file_uploader(
    "Upload Satellite (.nc) Files",
    type=["nc"],
    accept_multiple_files=True
)

live_file = st.sidebar.file_uploader(
    "Upload FAOSTAT CSV",
    type=["csv"]
)

if st.sidebar.button("Process Data"):

    if not sat_files:
        st.sidebar.error("Please upload at least one .nc satellite file.")
        st.stop()

    if not live_file:
        st.sidebar.error("Please upload the FAOSTAT CSV file.")
        st.stop()

    # --- Process Satellite ---
    records = []
    errors = []

    progress = st.sidebar.progress(0, text="Processing satellite files...")

    for i, file in enumerate(sat_files):
        try:
            # Write uploaded file to a temp file on disk (fixes xarray AxiosError with large files)
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            ds = xr.open_dataset(tmp_path, group="PRODUCT")
            ch4 = ds["methane_mixing_ratio_bias_corrected"]
            mean_val = float(np.nanmean(ch4.values))
            date_str = str(ds["time"].values[0])[:7]
            records.append([date_str, mean_val])
            ds.close()
            os.unlink(tmp_path)

        except Exception as e:
            errors.append(f"{file.name}: {str(e)}")
            # Clean up temp file if it exists
            try:
                os.unlink(tmp_path)
            except:
                pass
            continue

        progress.progress((i + 1) / len(sat_files), text=f"Processing file {i+1} of {len(sat_files)}...")

    progress.empty()

    if errors:
        with st.sidebar.expander(f"⚠️ {len(errors)} file(s) failed"):
            for e in errors:
                st.write(e)

    if not records:
        st.sidebar.error("No satellite files could be processed. Check file format.")
        st.stop()

    atmos_df = pd.DataFrame(records, columns=["Date", "Atmosphere_ppb"])
    atmos_df["Date"] = pd.to_datetime(atmos_df["Date"])
    atmos_df["Year"] = atmos_df["Date"].dt.year
    yearly_atmos = atmos_df.groupby("Year")["Atmosphere_ppb"].mean().reset_index()

    # --- Process Livestock ---
    try:
        livestock_df = pd.read_csv(live_file)
        livestock_df = livestock_df[["Area", "Year", "Value"]]
        livestock_df = livestock_df.rename(columns={"Value": "Livestock_kt"})
    except Exception as e:
        st.sidebar.error(f"Failed to read FAOSTAT CSV: {e}")
        st.stop()

    livestock_yearly = livestock_df.groupby("Year")["Livestock_kt"].sum().reset_index()

    merged_df = pd.merge(yearly_atmos, livestock_yearly, on="Year", how="inner")
    merged_df = merged_df.sort_values("Year").reset_index(drop=True)
    merged_df["Time_Index"] = range(len(merged_df))

    if merged_df.empty:
        st.sidebar.error("No overlapping years found between satellite and livestock data.")
        st.stop()

    st.session_state.merged_df = merged_df
    st.session_state.livestock_df = livestock_df
    st.session_state.yearly_atmos = yearly_atmos

    st.sidebar.success(f"✅ Processed {len(records)} satellite files successfully!")

# =====================================================
# PAGE CONTENT
# =====================================================

if "merged_df" in st.session_state:

    merged_df = st.session_state.merged_df
    livestock_df = st.session_state.livestock_df
    yearly_atmos = st.session_state.yearly_atmos

    # =================================================
    # PAGE 1 — GLOBAL MAPS
    # =================================================
    if st.session_state.page == 0:

        st.header("🌍 Global Methane Maps")

        year = st.selectbox("Select Year", merged_df["Year"])

        # Livestock Map
        year_data = livestock_df[livestock_df["Year"] == year]

        fig_livestock = px.scatter_geo(
            year_data,
            locations="Area",
            locationmode="country names",
            size="Livestock_kt",
            color="Livestock_kt",
            color_continuous_scale="Blues",
            projection="natural earth",
            title="Livestock Methane Emissions"
        )

        st.plotly_chart(fig_livestock, use_container_width=True)

        # Atmosphere Map
        atmos_value = yearly_atmos[yearly_atmos["Year"] == year]["Atmosphere_ppb"].values[0]

        atmos_map_df = pd.DataFrame({
            "Area": livestock_df["Area"].unique(),
            "Atmosphere_ppb": atmos_value
        })

        fig_atmos = px.scatter_geo(
            atmos_map_df,
            locations="Area",
            locationmode="country names",
            size="Atmosphere_ppb",
            color="Atmosphere_ppb",
            color_continuous_scale="Reds",
            projection="natural earth",
            title="Atmospheric Methane Concentration"
        )

        st.plotly_chart(fig_atmos, use_container_width=True)

        st.button("Next ➡️", on_click=next_page)

    # =================================================
    # PAGE 2 — CORRELATION SCATTER
    # =================================================
    elif st.session_state.page == 1:

        st.header("📊 Correlation Analysis")

        fig_scatter = px.scatter(
            merged_df,
            x="Livestock_kt",
            y="Atmosphere_ppb",
            trendline="ols",
            color_discrete_sequence=["purple"]
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        corr = merged_df["Livestock_kt"].corr(merged_df["Atmosphere_ppb"])
        st.metric("Correlation Coefficient", round(corr, 4))

        col1, col2 = st.columns(2)
        col1.button("⬅️ Previous", on_click=prev_page)
        col2.button("Next ➡️", on_click=next_page)

    # =================================================
    # PAGE 3 — HEATMAP
    # =================================================
    elif st.session_state.page == 2:

        st.header("🔥 Correlation Heatmap")

        fig_heat, ax = plt.subplots()
        sns.heatmap(
            merged_df[["Livestock_kt", "Atmosphere_ppb"]].corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig_heat)

        col1, col2 = st.columns(2)
        col1.button("⬅️ Previous", on_click=prev_page)
        col2.button("Next ➡️", on_click=next_page)

    # =================================================
    # PAGE 4 — TABLES
    # =================================================
    elif st.session_state.page == 3:

        st.header("📋 Statistical Summary")

        st.dataframe(merged_df.describe())

        col1, col2 = st.columns(2)
        col1.button("⬅️ Previous", on_click=prev_page)
        col2.button("Next ➡️", on_click=next_page)

    # =================================================
    # PAGE 5 — PREDICTION
    # =================================================
    elif st.session_state.page == 4:

        st.header("🤖 Future Prediction (Next 2 Years)")

        model_livestock = LinearRegression()
        model_atmos = LinearRegression()

        model_livestock.fit(merged_df[["Time_Index"]], merged_df["Livestock_kt"])
        model_atmos.fit(merged_df[["Time_Index"]], merged_df["Atmosphere_ppb"])

        future_years = [
            merged_df["Year"].max() + 1,
            merged_df["Year"].max() + 2
        ]

        future_index = range(len(merged_df), len(merged_df) + 2)

        future_livestock = model_livestock.predict(pd.DataFrame(future_index))
        future_atmos = model_atmos.predict(pd.DataFrame(future_index))

        future_df = pd.DataFrame({
            "Year": future_years,
            "Predicted_Livestock_kt": future_livestock,
            "Predicted_Atmosphere_ppb": future_atmos
        })

        st.dataframe(future_df)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=merged_df["Year"],
            y=merged_df["Livestock_kt"],
            name="Actual Livestock",
            line=dict(color="blue")
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_df["Year"],
            y=future_df["Predicted_Livestock_kt"],
            name="Predicted Livestock",
            line=dict(color="blue", dash="dash")
        ))
        fig_pred.add_trace(go.Scatter(
            x=merged_df["Year"],
            y=merged_df["Atmosphere_ppb"],
            name="Actual Atmosphere",
            line=dict(color="red")
        ))
        fig_pred.add_trace(go.Scatter(
            x=future_df["Year"],
            y=future_df["Predicted_Atmosphere_ppb"],
            name="Predicted Atmosphere",
            line=dict(color="red", dash="dash")
        ))

        st.plotly_chart(fig_pred, use_container_width=True)

        st.button("⬅️ Previous", on_click=prev_page)
