import streamlit as st
import pandas as pd
from utils.calculations1 import get_erlang_c
import plotly.graph_objects as go

# Set Streamlit page configuration for a wide layout
st.set_page_config(page_title="Agent Staffing Calculator", layout="wide")

# Load the static CSV file
@st.cache_data
def load_data():
    return pd.read_csv("forecast_test1.csv")




aht_seconds = 420  # 7 minutes

# Store the target SLA, max abandonment, shrinkage, etc., in session state for efficiency
if 'target_sla' not in st.session_state:
    st.session_state.target_sla = 80 / 100
    st.session_state.max_abandonment = 5
    st.session_state.shrinkage = 30 / 100
    st.session_state.caller_patience = 50
    st.session_state.target_answer_time = 60
    st.session_state.calculate_pressed = True  # Initially simulate the button press
    st.session_state.calculate_pressed1 = True  

def process_column(merged_df, column_name):
    results = []
    for volume in merged_df[column_name]:
        if pd.isna(volume) or volume < 1:  # Skip NaN or volumes < 1
            results.append({"No. of Required Agents w/ Shrinkage": 0, "SLA": 0, "pct of Calls Abandoned":0})
        else:
            traffic_intensity = (volume * aht_seconds) / 3600  # Traffic Intensity = volume * AHT in hours
            result = get_erlang_c(
                volume=volume,
                traffic_intensity=traffic_intensity,
                target_answer_time=st.session_state.target_answer_time,
                aht_seconds=aht_seconds,
                target_sla=st.session_state.target_sla,
                shrinkage=st.session_state.shrinkage,
                caller_patience=st.session_state.caller_patience,
                max_abandonment=st.session_state.max_abandonment
            )
            results.append(result)
    return results

def process_column1(merged_df1, column_name):
    results = []
    for volume in merged_df1[column_name]:
        if pd.isna(volume) or volume < 1:  # Skip NaN or volumes < 1
            results.append({"No. of Required Agents w/ Shrinkage": 0, "SLA": 0, "pct of Calls Abandoned":0})
        else:
            traffic_intensity = (volume * aht_seconds) / 3600  # Traffic Intensity = volume * AHT in hours
            result = get_erlang_c(
                volume=volume,
                traffic_intensity=traffic_intensity,
                target_answer_time=st.session_state.target_answer_time1,
                aht_seconds=aht_seconds,
                target_sla=st.session_state.target_sla1,
                shrinkage=st.session_state.shrinkage1,
                caller_patience=st.session_state.caller_patience1,
                max_abandonment=st.session_state.max_abandonment1
            )
            results.append(result)
    return results

# Load data
merged_df = load_data()
merged_df1 = load_data()
merged_df['index'] = pd.to_datetime(merged_df['index'], errors='coerce')
merged_df1['index'] = pd.to_datetime(merged_df1['index'], errors='coerce')


tab1, tab3 = st.tabs(["Inbound (Prediction Tool)", "Actual vs Optimal vs Prediction"])

def plot_forecast_agents_by_hour(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted # Of Agents"], mode='lines', name='Predicted # Of Agents'))
    fig.add_trace(go.Scatter(x=df["index"], y=df["Lower Bound (# Of Agents)"], mode='lines', name='Lower Bound (# Of Agents)'))
    fig.add_trace(go.Scatter(x=df["index"], y=df["Upper Bound (# Of Agents)"], mode='lines', name='Upper Bound (# Of Agents)'))
    fig.update_layout(
        title="Forecast Agents by Hour",
        xaxis_title="Hour",
        yaxis_title="# Of Agents",
        height=500,  # Set height for better visualization
        template="plotly_white",  # Keeping the cooler theme consistent
    )
    return fig

def plot_forecast_agents_by_hour(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["index"], 
        y=df["Predicted # Of Agents"], 
        name='Predicted # Of Agents', 
        marker_color='blue'  # Set color for bars
    ))
    fig.add_trace(go.Bar(
        x=df["index"], 
        y=df["Lower Bound (# Of Agents)"], 
        name='Lower Bound (# Of Agents)', 
        marker_color='green'  # Set color for bars
    ))
    fig.add_trace(go.Bar(
        x=df["index"], 
        y=df["Upper Bound (# Of Agents)"], 
        name='Upper Bound (# Of Agents)', 
        marker_color='orange'  # Set color for bars
    ))
    fig.update_layout(
        barmode='group',  # Group the bars together
        title="Forecast Agents by Hour",
        xaxis_title="Hour",
        yaxis_title="# Of Agents",
        height=500,  # Set height for better visualization
        template="plotly_white",  # Keeping the cooler theme consistent
    )
    return fig

with tab1:
    # Streamlit layout
    st.title("Contact Centre Staffing Intelligence")

    # Ensure the chart spans across the entire page width
    def plot_forecast_interactions_by_hour(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted Volume"], mode='lines', name='Predicted Volume')) 
        fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted Lower Bound"], mode='lines', name='Lower Bound'))
        fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted Upper Bound"], mode='lines', name='Upper Bound'))
        fig.update_layout(
            title="Forecast Interactions by Hour",
            xaxis_title="Hour",
            yaxis_title="Volume",
            height=500,  # Set height for better visualization
         )
        return fig

    # Display the forecast interaction chart
    st.plotly_chart(plot_forecast_interactions_by_hour(merged_df), use_container_width=True)

    # Create two columns for input fields and data preview
    left_column, right_column = st.columns([3, 1])

    with right_column:

        date_range = st.date_input(
            "Date Range",
            value=(merged_df["index"].min().date(), merged_df["index"].max().date()),
            help="Pick a start and end date to filter the data."
        )

        if len(date_range) == 2:
            start_date, end_date = date_range

            # Ensure the selected range is valid
            if start_date > end_date:
                st.error("Start date cannot be after end date.")
            else:
                start_timestamp = pd.Timestamp(start_date)
                end_timestamp = pd.Timestamp(end_date)
                # Handle the case where the start and end date are the same
                merged_df = merged_df[
                    (merged_df["index"].dt.date >= start_timestamp.date()) &
                    (merged_df["index"].dt.date <= end_timestamp.date())
                ]

        else:
            st.warning("Please select a valid date range.")



        # Input Fields
        st.session_state.max_abandonment = st.slider("Max. Abandonment (as %)", min_value=0, max_value=20, value=5)
        st.session_state.target_sla = st.slider("Target SLA (as %)", min_value=50, max_value=99, value=80) / 100
        st.session_state.target_answer_time = st.number_input("Target Answer Time (in seconds)", min_value=1, step=1, value=60)
        st.session_state.shrinkage = st.slider("Shrinkage (as %)", min_value=0, max_value=100, value=30) / 100
        st.session_state.caller_patience = st.number_input("Enter Caller Patience Level (in seconds)", min_value=1, step=1, value=50)

        # Button to calculate
        if st.session_state.calculate_pressed or st.button("Calculate Required Staff"):
            # Reset session state so that calculations do not repeat unless the button is clicked again
            st.session_state.calculate_pressed = False

            # Perform the calculations
            columns_to_process = ["Actual Volume", "Predicted Volume", "Predicted Lower Bound", "Predicted Upper Bound"]
            for column in columns_to_process:
                processed_results = process_column(merged_df, column)
                merged_df[f"{column}_agents_with_shrinkage"] = [res["No. of Required Agents w/ Shrinkage"] for res in processed_results]
                merged_df[f"{column}_sla"] = [res["SLA"] for res in processed_results]
                merged_df[f"{column}_max_abd"] = [res["pct of Calls Abandoned"] for res in processed_results]

            # Rename the columns for consistency
            merged_df.rename(columns={
                "Predicted Lower Bound": "Lower Bound (Volume)",
                "Predicted Upper Bound": "Upper Bound (Volume)",
                "Actual Volume_agents_with_shrinkage": "# Of Agents (Actually Needed)",
                "Predicted Volume_agents_with_shrinkage": "Predicted # Of Agents",
                "Predicted Lower Bound_agents_with_shrinkage": "Lower Bound (# Of Agents)",
                "Predicted Upper Bound_agents_with_shrinkage": "Upper Bound (# Of Agents)"
            }, inplace=True)

            # Save the calculated results in session_state
            st.session_state.merged_df = merged_df

    with left_column:
        # Select specific columns to display
        selected_columns = ["index","Predicted Volume", "Lower Bound (Volume)", "Upper Bound (Volume)", "Predicted # Of Agents", "Lower Bound (# Of Agents)", "Upper Bound (# Of Agents)"]

        # Check if calculated columns exist before displaying
        available_columns = [col for col in selected_columns if col in merged_df.columns]

        st.markdown("**Preview of Selected Forecast Data:**")
        if available_columns:
            st.dataframe(merged_df[available_columns])
        else:
            st.warning("Calculated columns are not available. Please click 'Calculate Required Staff' to generate them.")


    # Check if the button to calculate was pressed
    if not st.session_state.calculate_pressed:


        # Check for required columns for Agents chart
        required_columns_agents = [
            "Predicted # Of Agents",
            "Lower Bound (# Of Agents)",
            "Upper Bound (# Of Agents)"
        ]
        if all(col in st.session_state.merged_df.columns for col in required_columns_agents):
            st.plotly_chart(plot_forecast_agents_by_hour(st.session_state.merged_df), use_container_width=True)
        else:
            st.warning("Required columns for the Agents chart are not available. Please click 'Calculate Required Staff' to generate them.")


    left_column, right_column = st.columns(2)


    with left_column:
        # After the button is clicked, plot the chart for SLA
        if not st.session_state.calculate_pressed:
            if 'Predicted Volume_sla' in st.session_state.merged_df.columns:
                def plot_predicted_sla(df):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted Volume_sla"], mode='lines', name='Predicted SLA'))
                    fig.update_layout(
                        title="Predicted SLA",
                        xaxis_title="Hour",
                        yaxis_title="SLA",
                        height=500,  # Set height for better visualization
                    )
                    return fig

                st.plotly_chart(plot_predicted_sla(st.session_state.merged_df))
            else:
                st.warning("The predicted SLA data is not available. Please click 'Calculate Required Staff' to generate it.")

    with right_column:
        # After the button is clicked, plot the chart for SLA
        if not st.session_state.calculate_pressed:
            if 'Predicted Volume_max_abd' in st.session_state.merged_df.columns:
                def plot_predicted_max_abd(df):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted Volume_max_abd"], mode='lines', name='Predicted Max Abandonment %'))
                    fig.update_layout(
                        title="Predicted Max Abandonment %",
                        xaxis_title="Hour",
                        yaxis_title="Max Abandonment %",
                        height=500,  # Set height for better visualization
                    )
                    return fig

                st.plotly_chart(plot_predicted_max_abd(st.session_state.merged_df))
            else:
                st.warning("The predicted Abadonment  data is not available. Please click 'Calculate Required Staff' to generate it.")





with tab3:
    # Streamlit layout
    st.title("Actual vs Optimal vs Prediction for Agent Staffing")
    left_column, right_column = st.columns([4, 1])

    with right_column:

        date_range1 = st.date_input(
            "Date Range",
            value=(merged_df1["index"].min().date(), merged_df1["index"].max().date()),
            help="Pick a start and end date to filter the data.",
            key = "date_range_1"
        )

        if len(date_range1) == 2:
            start_date1, end_date1 = date_range1

            # Ensure the selected range is valid
            if start_date1 > end_date1:
                st.error("Start date cannot be after end date.")
            else:
                start_timestamp1 = pd.Timestamp(start_date1)
                end_timestamp1 = pd.Timestamp(end_date1)
                # Handle the case where the start and end date are the same
                merged_df1 = merged_df1[
                    (merged_df1["index"].dt.date >= start_timestamp1.date()) &
                    (merged_df1["index"].dt.date <= end_timestamp1.date())
                ]

        else:
            st.warning("Please select a valid date range.")





        # Input Fields
        st.session_state.max_abandonment1 = st.slider("Max. Abandonment (as %)", min_value=0, max_value=20, value=5, key = "max_abandonment_1")
        st.session_state.target_sla1 = st.slider("Target SLA (as %)", min_value=50, max_value=99, value=80, key ="target_sla_1") / 100
        st.session_state.target_answer_time1 = st.number_input("Target Answer Time (in seconds)", min_value=1, step=1, value=60, key="target_answer_time_1")
        st.session_state.shrinkage1 = st.slider("Shrinkage (as %)", min_value=0, max_value=100, value=30, key = "shrinkage_1") / 100
        st.session_state.caller_patience1 = st.number_input("Enter Caller Patience Level (in seconds)", min_value=1, step=1, value=50, key = "caller_patience_1")

        # Button to calculate
        if st.session_state.calculate_pressed1 or st.button("Calculate Comparison"):
            # Reset session state so that calculations do not repeat unless the button is clicked again
            st.session_state.calculate_pressed1 = False

            # Perform the calculations
            columns_to_process = ["Actual Volume", "Predicted Volume", "Predicted Lower Bound", "Predicted Upper Bound"]
            for column in columns_to_process:
                processed_results1 = process_column1(merged_df1, column)
                merged_df1[f"{column}_agents_with_shrinkage"] = [res["No. of Required Agents w/ Shrinkage"] for res in processed_results1]
                merged_df1[f"{column}_sla"] = [res["SLA"] for res in processed_results1]
                merged_df1[f"{column}_max_abd"] = [res["pct of Calls Abandoned"] for res in processed_results1]

            # Rename the columns for consistency
            merged_df1.rename(columns={
                "Predicted Lower Bound": "Lower Bound (Volume)",
                "Predicted Upper Bound": "Upper Bound (Volume)",
                "Actual Volume_agents_with_shrinkage": "# Of Agents (Actually Needed)",
                "Predicted Volume_agents_with_shrinkage": "Predicted # Of Agents",
                "Predicted Lower Bound_agents_with_shrinkage": "Lower Bound (# Of Agents)",
                "approx_inbound_agents": "Actual Inbound Agents",
                "Predicted Upper Bound_agents_with_shrinkage": "Upper Bound (# Of Agents)"
            }, inplace=True)

            # Save the calculated results in session_state
            st.session_state.merged_df1 = merged_df1

    with left_column:
        
        selected_columns = ["index","Actual Volume","Predicted Volume","Actual Inbound Agents","# Of Agents (Actually Needed)", "Predicted # Of Agents", "Lower Bound (# Of Agents)", "Upper Bound (# Of Agents)"]

        available_columns = [col for col in selected_columns if col in merged_df1.columns]

        st.markdown("**Preview of Data:**")
        if available_columns:
            st.dataframe(merged_df1[available_columns])
        else:
            st.warning("Calculated columns are not available. Please click 'Calculate Required Staff' to generate them.")



        #if not merged_df1.empty:
        #    st.dataframe(merged_df1[selected_columns])  # Display the entire DataFrame
        #else:
        #    st.warning("The forecast data is not available. Please click 'Calculate Required Staff' to generate it.")

        # Ensure the chart spans across the entire page width
    def plot_agents_interactions_comparison(df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["index"], y=df["Actual Inbound Agents"], mode='lines', name='Actual Inbound Agents'))
        fig.add_trace(go.Scatter(x=df["index"], y=df["Predicted # Of Agents"], mode='lines', name='Predicted # Of Agents'))
        fig.add_trace(go.Scatter(x=df["index"], y=df["# Of Agents (Actually Needed)"], mode='lines', name='Optimal # Of Agents'))
        fig.add_trace(go.Scatter(x=df["index"], y=df["Lower Bound (# Of Agents)"], mode='lines', name='Lower Bound Prediction'))
        fig.add_trace(go.Scatter(x=df["index"], y=df["Upper Bound (# Of Agents)"], mode='lines', name='Upper Bound Prediction'))
        fig.update_layout(
            title="Forecast Interactions by Hour",
            xaxis_title="Hour",
            yaxis_title="Volume",
            height=500,  # Set height for better visualization
        )
        return fig

    if not st.session_state.calculate_pressed1:

        required_columns_agents = [
            "Predicted # Of Agents",
            "Lower Bound (# Of Agents)",
            "Upper Bound (# Of Agents)",
            "# Of Agents (Actually Needed)",
            "Actual Inbound Agents"
        ]
        if all(col in st.session_state.merged_df1.columns for col in required_columns_agents):
            st.plotly_chart(plot_agents_interactions_comparison(st.session_state.merged_df1), use_container_width=True)
        else:
            st.warning("Required columns for the Agents chart are not available. Please click 'Calculate Required Staff' to generate them.")

    col1, col2, col3 = st.columns(3)

    with col1:    
        def calculate_and_display_model_kpi(merged_df1):
            if merged_df1.empty:
                st.warning("No data available to calculate metrics.")
                return

            # Ensure required columns exist
            required_columns = ["Predicted # Of Agents", "Actual Inbound Agents", "# Of Agents (Actually Needed)"]
            if not all(col in merged_df1.columns for col in required_columns):
                st.error("The required columns are missing from the data. Please check the input.")
                return

            # Step 1: Calculate absolute differences
            merged_df1["model_diff"] = abs(merged_df1["Predicted # Of Agents"] - merged_df1["# Of Agents (Actually Needed)"])
            merged_df1["actual_diff"] = abs(merged_df1["Actual Inbound Agents"] - merged_df1["# Of Agents (Actually Needed)"])

            # Step 2: Determine which is closer
            merged_df1["better_prediction"] = merged_df1["model_diff"] <= merged_df1["actual_diff"]
            model_wins = merged_df1["better_prediction"].sum()  # Count where model predicted closer
            total_predictions = len(merged_df1)
            actual_wins = total_predictions-model_wins
            print(actual_wins)
            print(model_wins)


            # Step 3: Calculate metric
            accuracy_metric = ((model_wins-actual_wins) / total_predictions) * 100

            # Step 4: Display KPI tile as Plotly figure
            fig = go.Figure()

            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=accuracy_metric,
                    number={"suffix": "%", "font": {"size": 48}},
                    title={"text": "Model Outperforming Actual By"},
                    domain={"x": [0, 1], "y": [0, 1]},
                )
            )

            fig.update_layout(
                height=250,  # Set height for tile
                margin=dict(t=0, b=0, l=0, r=0),  # Remove margins for a clean look
            )

            st.plotly_chart(fig, use_container_width=True)







        if not st.session_state.calculate_pressed1:
            calculate_and_display_model_kpi(st.session_state.merged_df1)

    with col2:

        def calculate_and_display_agent_hours_saved(merged_df1):
            if merged_df1.empty:
                st.warning("No data available to calculate agent hours saved.")
                return

            # Ensure required columns exist
            required_columns = ["Predicted # Of Agents", "Actual Inbound Agents", "# Of Agents (Actually Needed)", "model_diff", "actual_diff", "better_prediction"]
            if not all(col in merged_df1.columns for col in required_columns):
                st.error("The required columns are missing from the data. Please check the input.")
                return

            # Step 1: Filter rows where the model won and predicted higher or equal
            savings_condition = (
                (merged_df1["better_prediction"]) &  # Model won
                (merged_df1["Predicted # Of Agents"] >= merged_df1["# Of Agents (Actually Needed)"])
            )
            savings_rows = merged_df1[savings_condition]

            # Step 2: Calculate agent hours saved
            savings_rows["agent_hours_saved"] = savings_rows.apply(
                lambda row: max(row["Actual Inbound Agents"] - row["Predicted # Of Agents"], 0),
                axis=1
            )
            total_hours_saved = savings_rows["agent_hours_saved"].sum()

            # Step 3: Display KPI tile as Plotly figure
            fig = go.Figure()

            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=total_hours_saved,
                    number={"font": {"size": 48}},
                    title={"text": "Total Agent Hours Saved"},
                    domain={"x": [0, 1], "y": [0, 1]},
                )
            )

            fig.update_layout(
                height=250,  # Set height for tile
                margin=dict(t=0, b=0, l=0, r=0),  # Remove margins for a clean look
            )
            

            st.plotly_chart(fig, use_container_width=True)

            

        # Call the KPI functions in the Streamlit app
        if not st.session_state.calculate_pressed1:
            
            calculate_and_display_agent_hours_saved(st.session_state.merged_df1)  # New KPI
    

    with col3:
        def calculate_and_display_agent_hours_saved(merged_df1):
            if merged_df1.empty:
                st.warning("No data available to calculate agent hours saved.")
                return None, 0

            required_columns = ["Predicted # Of Agents", "Actual Inbound Agents", "# Of Agents (Actually Needed)", "model_diff", "actual_diff", "better_prediction"]
            if not all(col in merged_df1.columns for col in required_columns):
                st.error("The required columns are missing from the data. Please check the input.")
                return None, 0

            savings_condition = (
                (merged_df1["better_prediction"]) & 
                (merged_df1["Predicted # Of Agents"] >= merged_df1["# Of Agents (Actually Needed)"])
            )
            savings_rows = merged_df1[savings_condition]
            savings_rows["agent_hours_saved"] = savings_rows.apply(
                lambda row: max(row["Actual Inbound Agents"] - row["Predicted # Of Agents"], 0),
                axis=1
            )
            total_hours_saved = savings_rows["agent_hours_saved"].sum()

            fig = go.Figure()
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=total_hours_saved,
                    number={"font": {"size": 48}},
                    title={"text": "Total Agent Hours Saved"},
                    domain={"x": [0, 1], "y": [0, 1]},
                )
            )
            fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
            return fig, total_hours_saved

        def calculate_and_display_cost_savings(total_hours_saved):
            cost_per_hour1 = st.number_input(
                label="Enter cost per hour ($):",
                min_value=10,
                max_value=200,
                value=50,
                step=1,
                key="eedc"
            )
            total_dollar_savings = total_hours_saved * cost_per_hour1

            fig = go.Figure()
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=total_dollar_savings,
                    number={"prefix": "$", "font": {"size": 48}},
                    title={"text": "Total Cost Savings"},
                    domain={"x": [0, 1], "y": [0, 1]},
                )
            )
            fig.update_layout(height=250, margin=dict(t=0, b=0, l=0, r=0))
            return fig

        if not st.session_state.calculate_pressed1:

            agent_hours_fig, total_hours_saved = calculate_and_display_agent_hours_saved(st.session_state.merged_df1)
            
            cost_savings_fig = calculate_and_display_cost_savings(total_hours_saved)
            st.plotly_chart(cost_savings_fig, use_container_width=True)







