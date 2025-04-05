import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned data
@st.cache_data
def load_data():
    df = pd.read_excel("Instructional Hours Data Request - School Year 2024-25 - 4-3-2025.xlsx", sheet_name="Sheet1")
    df = df[['RPTING_LEA_IRN', 'RPTING_LEA_NAME', 'TOTAL_SCHOOL_YEAR_HOURS_COUNT']].dropna()
    df = df.groupby(['RPTING_LEA_IRN', 'RPTING_LEA_NAME'])['TOTAL_SCHOOL_YEAR_HOURS_COUNT'].mean().reset_index()
    df.columns = ['District IRN', 'District Name', 'Avg Instructional Hours']
    return df

data = load_data()

# Title
st.title("Ohio School District Instructional Hours (2024–25)")

# Histogram with quartiles
st.subheader("Distribution of Average Instructional Hours")
fig1, ax1 = plt.subplots()
sns.histplot(data['Avg Instructional Hours'], bins=30, kde=True, ax=ax1)
ax1.axvline(data['Avg Instructional Hours'].mean(), color='red', linestyle='--', label='Mean')
ax1.axvline(data['Avg Instructional Hours'].median(), color='green', linestyle='--', label='Median')
ax1.set_xlabel("Average Instructional Hours")
ax1.set_ylabel("Number of Districts")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Top and Bottom 10 Districts
st.subheader("Top 10 and Bottom 10 Districts by Instructional Hours")
top10 = data.sort_values(by='Avg Instructional Hours', ascending=False).head(10)
bottom10 = data.sort_values(by='Avg Instructional Hours').head(10)

fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.barplot(data=top10, y='District Name', x='Avg Instructional Hours', ax=ax3)
ax3.set_title("Top 10 Districts")
ax3.set_xlabel("Average Instructional Hours")
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(data=bottom10, y='District Name', x='Avg Instructional Hours', ax=ax4)
ax4.set_title("Bottom 10 Districts")
ax4.set_xlabel("Average Instructional Hours")
st.pyplot(fig4)

# Downloadable Data Table
st.subheader("District Instructional Hours Table")
search = st.text_input("Search for a district name")
filtered = data[data['District Name'].str.contains(search, case=False, na=False)]
filtered = filtered.sort_values(by='Avg Instructional Hours', ascending=False)

# Conditional formatting
def highlight_rows(row):
    if row['Avg Instructional Hours'] < 1001:
        return ['background-color: red'] * len(row)
    elif 1001 <= row['Avg Instructional Hours'] <= 1054:
        return ['background-color: orange'] * len(row)
    else:
        return [''] * len(row)

styled_df = filtered.style.apply(highlight_rows, axis=1)
st.write(styled_df.to_html(escape=False), unsafe_allow_html=True)

csv = filtered.to_csv(index=False)
st.download_button(
    "Download CSV",
    csv,
    "instructional_hours_by_district.csv",
    "text/csv",
    key='download-csv'
)
