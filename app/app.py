import streamlit as st
from hypotheses import hypothesis1, hypothesis2, hypothesis3, hypothesis4
from streamlit_helper import display_and_modify_table
from preprocess import preprocess_data
from dbcrud import upload_data_in_db

st.set_page_config(
    page_title="Olympic Trends Analysis",
    page_icon=":book:",
    layout="centered",
    initial_sidebar_state="auto"
)

def safe_execute(func, error_message, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"{error_message}: {str(e)}")
        return None

if "is_running" not in st.session_state:
    st.session_state.is_running = False
    st.session_state.running_hypothesis = None

def start_running(hypothesis_name):
    st.session_state.is_running = True
    st.session_state.running_hypothesis = hypothesis_name

def stop_running():
    st.session_state.is_running = False
    st.session_state.running_hypothesis = None

st.title("Olympic Trends Analysis")

st.header("Upload from Raw CSV files")
st.warning("This will overwrite all existing data if it exists. Do this if you believe the tables have been modified incorrectly.")
if st.button("Upload"):
    st.info("Uploading...")
    safe_execute(upload_data_in_db, "Failed to upload data to the database.")
    st.info("Data has been successfully populated in Raw Tables")

st.header("Available raw tables")
tables = ["Athlete_Events_Details", "Event_Results", "Athlete_Biography",
          "Medal_Tally", "Games_Summary", "Population_Total", "Country_Profile"]

selected_table = st.selectbox("Select Table to View/Modify", tables)
if selected_table:
    safe_execute(display_and_modify_table, f"Failed to display or modify table: {selected_table}", st, selected_table)

st.header("Preprocessing Steps")
st.info("These will do preprocessing from Raw Data in the database and create new tables.")
st.warning("This will truncate preprocessed tables.")

if st.button("Run Preprocessing"):
    safe_execute(preprocess_data, "Failed to run preprocessing steps.")


st.header("Available Processed Tables")
processed_tables = ["Pre_Event_Results", "Pre_Population_Total", "Pre_Athlete_Biography", 
                    "Pre_Athlete_Events_Details", "Pre_Country_Profile"]

selected_processed_table = st.selectbox("Select Processed Table to View/Modify", processed_tables)
if selected_processed_table:
    safe_execute(display_and_modify_table, f"Failed to display or modify processed table: {selected_processed_table}", st, selected_processed_table)

st.header("Olympic Trends Analysis")
st.write("""
### Hypotheses
1. **What is the general trend in women participation country wise over the years? What countries are doing well and how do they compare to the best performing countries?**
2. **Are there any sports which are on the decline and losing popularity among participants? Also, are there some sports which have gained popularity over the recent years?**
3. **How do the trends in medal counts for team sports compare to those for individual sports across different countries over the years, and what insights can be drawn from these comparisons regarding each country's performance in the Olympic Games?**
4. **Table Tennis and Tennis are similar yet different sports. The players we have seen in both games seem to have different builds. The hypothesis is that we can build a model using Height, Weight, and athlete’s country to predict which sport they belong to.**
""")

st.header("Hypothesis 1")
st.write("What is the general trend in women participation country wise over the years? What countries are doing well and how do they compare to the best performing countries?")
if st.button("Run Hypothesis 1", disabled=st.session_state.is_running):
    start_running("Hypothesis 1")
    try:
        hypothesis1(st)
        st.success("Hypothesis 1 executed successfully!")
    except Exception as e:
        st.error(f"Error while running Hypothesis 1: {str(e)}")
    finally:
        stop_running()

st.header("Hypothesis 2")
st.write("Are there any sports which are on the decline and losing popularity among participants? Also, are there some sports which have gained popularity over the recent years?")
if st.button("Run Hypothesis 2", disabled=st.session_state.is_running):
    start_running("Hypothesis 2")
    try:
        hypothesis2(st)
        st.success("Hypothesis 2 executed successfully!")
    except Exception as e:
        st.error(f"Error while running Hypothesis 2: {str(e)}")
    finally:
        stop_running()

st.header("Hypothesis 3")
st.write("How do the trends in medal counts for team sports compare to those for individual sports across different countries over the years, and what insights can be drawn from these comparisons regarding each country's performance in the Olympic Games?")
if st.button("Run Hypothesis 3", disabled=st.session_state.is_running):
    start_running("Hypothesis 3")
    try:
        hypothesis3(st)
        st.success("Hypothesis 3 executed successfully!")
    except Exception as e:
        st.error(f"Error while running Hypothesis 3: {str(e)}")
    finally:
        stop_running()

st.header("Hypothesis 4")
st.write("Table Tennis and Tennis are similar yet different sports. The players we have seen in both games seem to have different builds. The hypothesis is that we can build a model using Height, Weight, and athlete’s country to predict which sport they belong to.")
if st.button("Run Hypothesis 4", disabled=st.session_state.is_running):
    start_running("Hypothesis 4")
    try:
        hypothesis4(st)
        st.success("Hypothesis 4 executed successfully!")
    except Exception as e:
        st.error(f"Error while running Hypothesis 4: {str(e)}")
    finally:
        stop_running()