import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the cleaned instructional hours data
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

st.markdown("""
### Key Takeaways

One of the central hypotheses explored in this analysis was whether increasing instructional hours has a measurable impact on student performance. The regression models and scatter plots indicate that there is indeed a positive relationship, though not overwhelmingly strong. The data suggest that student performance, as measured by the Performance Index, tends to improve as instructional time increases — up to a point. The quadratic regression model estimates that **optimal instructional hours are around 1,167 hours per year**, a figure higher than what most Ohio districts currently offer.

However, when we examine which districts are actually able to provide over 1,200 hours of instruction, a clear pattern emerges: they are overwhelmingly **small** districts. Larger districts, particularly those serving more than 10,000 students, are almost completely absent from the list of high-hour systems. This limitation among large districts is unlikely to be incidental. Instead, it likely reflects **structural and contractual constraints**, including limits set by **collective bargaining agreements (CBAs)** that define teacher work hours, prep time, and student contact limits.

This insight is reinforced by visualizations comparing instructional time to district enrollment, where it becomes clear that no very large public district exceeds the 1,200-hour threshold. These findings suggest that efforts to expand instructional time across the state — particularly in large urban systems — will require a **collaborative approach**. This includes working closely with **district leadership and teaching unions** to explore options that balance instructional quality with sustainable working conditions.

Moreover, when comparing public districts with more than 1,200 hours to those with fewer, we see mixed results on factors like **chronic absenteeism**, **income**, and **expenditures per student**. While high-hour districts sometimes outperform, it's clear that **time alone isn't the only factor** — but it may be a necessary one. In combination with strategic supports and funding, extending the school year could be an important tool for improving outcomes, particularly in high-poverty or high-need areas.
""")

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

# Load full dataset with predictors
@st.cache_data
def load_full_regression_data():
    avg_hours_df = load_data()
    performance_df = pd.read_csv("Performance_Index-Table 1.csv")
    overview_df = pd.read_excel("DISTRICT_HIGH_LEVEL_2324 (1).xlsx", sheet_name="DISTRICT_OVERVIEW")
    edchoice_df = pd.read_excel("23-24_District_School_Options (2).xlsx", sheet_name="District School Options")
    chronic_df = pd.read_excel("2024_District_Details (2).xlsx", sheet_name="District_Details")
    income_df = pd.read_csv("income_districts.csv")

    income_df['Match Name'] = income_df['School district'].str.lower().str.replace(" city| local| exempted village| school district| schools", "", regex=True).str.strip()
    performance_df['Match Name'] = performance_df['District Name'].str.lower().str.replace(" city| local| exempted village| school district| schools", "", regex=True).str.strip()
    agi_col = [col for col in income_df.columns if 'agi' in col.lower()][0]
    income_df.rename(columns={agi_col: 'Average AGI'}, inplace=True)
    income_df['Average AGI'] = income_df['Average AGI'].replace('[\$,]', '', regex=True).astype(float)

    merged = pd.merge(performance_df, income_df[['Match Name', 'Average AGI']], on='Match Name', how='left')
    merged = pd.merge(merged, avg_hours_df, on='District IRN', how='inner')
    edchoice_df['EdChoice Count'] = pd.to_numeric(edchoice_df['Number of EdChoice Expansion Scholarship Students 2023-2024'].replace('<10', '5'), errors='coerce')
    merged = pd.merge(merged, edchoice_df[['District IRN', 'EdChoice Count']], on='District IRN', how='inner')
    chronic_clean = chronic_df[chronic_df['Student Group'] == 'All Students'][['District IRN', 'Chronic Absenteeism Rate']]
    chronic_clean['Chronic Absenteeism Rate'] = pd.to_numeric(chronic_clean['Chronic Absenteeism Rate'], errors='coerce')
    merged = pd.merge(merged, chronic_clean, on='District IRN', how='inner')

    merged['Expenditures'] = merged['Expenditures per Equivalent Pupil - State and Local Funds'].replace('[\$,]', '', regex=True).astype(float)
    merged['Teachers_FTE'] = pd.to_numeric(merged['Number of General Education Teachers FTE'], errors='coerce')

    return merged.dropna(subset=['Performance Index Score 2023-2024', 'Avg Instructional Hours', 'Teachers_FTE', 'Expenditures', 'Average AGI', 'EdChoice Count', 'Chronic Absenteeism Rate'])

# Quadratic regression chart
st.subheader("Quadratic Fit: Instructional Hours vs Performance Index")
reg_df = load_full_regression_data().copy()
reg_df['Hours^2'] = reg_df['Avg Instructional Hours'] ** 2
Xq = reg_df[['Avg Instructional Hours', 'Hours^2', 'Teachers_FTE', 'Expenditures', 'Average AGI', 'EdChoice Count', 'Chronic Absenteeism Rate']]
Xq = sm.add_constant(Xq)
yq = reg_df['Performance Index Score 2023-2024']
model_q = sm.OLS(yq, Xq).fit()

b1 = model_q.params['Avg Instructional Hours']
b2 = model_q.params['Hours^2']
optimal_hours = -b1 / (2 * b2)

st.markdown(f"**Estimated optimal instructional hours:** {optimal_hours:.0f} hours")

hour_range = pd.Series(range(int(reg_df['Avg Instructional Hours'].min()), int(reg_df['Avg Instructional Hours'].max()) + 1))
y_pred = model_q.params['const'] + b1 * hour_range + b2 * hour_range ** 2

fig_opt, ax_opt = plt.subplots()
ax_opt.plot(hour_range, y_pred, label='Predicted Performance', color='blue')
ax_opt.axvline(optimal_hours, color='red', linestyle='--', label=f'Optimal = {optimal_hours:.0f} hrs')
ax_opt.set_xlabel("Instructional Hours")
ax_opt.set_ylabel("Predicted Performance Index")
ax_opt.set_title("Performance vs Instructional Hours (Quadratic Fit)")
ax_opt.legend()
ax_opt.grid(True)
st.pyplot(fig_opt)



# Scatterplot: Instructional Hours vs Chronic Absenteeism
st.subheader("Instructional Hours vs Chronic Absenteeism")
fig_abs, ax_abs = plt.subplots()
sns.scatterplot(data=reg_df, x='Avg Instructional Hours', y='Chronic Absenteeism Rate', ax=ax_abs)
ax_abs.axhline(reg_df['Chronic Absenteeism Rate'].mean(), color='red', linestyle='--', label='Mean Absenteeism')
ax_abs.set_xlabel("Average Instructional Hours")
ax_abs.set_ylabel("Chronic Absenteeism Rate")
ax_abs.set_title("Instructional Hours vs Chronic Absenteeism")
ax_abs.legend()
ax_abs.grid(True)
st.pyplot(fig_abs)

# Scatterplot: Actual vs Predicted Performance Index
st.subheader("Actual vs Predicted Performance Index")
reg_df['Predicted PI'] = model_q.predict(Xq)

fig_pred, ax_pred = plt.subplots()
sns.scatterplot(x=reg_df['Predicted PI'], y=reg_df['Performance Index Score 2023-2024'], ax=ax_pred)
ax_pred.plot([reg_df['Predicted PI'].min(), reg_df['Predicted PI'].max()],
             [reg_df['Predicted PI'].min(), reg_df['Predicted PI'].max()],
             color='red', linestyle='--', label='Perfect Prediction')
ax_pred.set_xlabel("Predicted Performance Index")
ax_pred.set_ylabel("Actual Performance Index")
ax_pred.set_title("Actual vs Predicted Performance Index")
ax_pred.legend()
ax_pred.grid(True)
st.pyplot(fig_pred)

# Standard multi-variable regression summary output

# Scatterplot: Rural vs Instructional Hours

# Boxplot: Instructional Hours by Public vs Charter/Private
st.subheader("Instructional Hours by District Type (Public vs Charter/Private)")
performance_df = pd.read_csv("Performance_Index-Table 1.csv")
performance_df['Public Flag'] = 1

# Merge with main data to label public districts
pub_merge = data.merge(performance_df[['District IRN', 'Public Flag']], on='District IRN', how='left')
pub_merge['District Type'] = pub_merge['Public Flag'].fillna(0).replace({1: 'Public', 0: 'Charter/Private'})

fig_type, ax_type = plt.subplots()
sns.boxplot(data=pub_merge, x='District Type', y='Avg Instructional Hours', ax=ax_type)
ax_type.set_title("Average Instructional Hours: Public vs Charter/Private")
ax_type.set_xlabel("District Type")
ax_type.set_ylabel("Average Instructional Hours")
st.pyplot(fig_type)

# Analyze public districts with more than 1200 instructional hours
st.subheader("Public Districts Offering More Than 1200 Instructional Hours")
public_1200_plus = pub_merge[(pub_merge['District Type'] == 'Public') & (pub_merge['Avg Instructional Hours'] > 1200)]
st.write(public_1200_plus.sort_values(by='Avg Instructional Hours', ascending=False))

# Compare public 1200+ to all public districts
public_all = pub_merge[pub_merge['District Type'] == 'Public']
public_1200_plus_ids = public_1200_plus['District IRN'].tolist()
comparison_df = reg_df[reg_df['District IRN'].isin(public_1200_plus_ids)].copy()
comparison_df['Group'] = '1200+ Hours'

other_public_df = reg_df[(reg_df['District IRN'].isin(public_all['District IRN'])) & (~reg_df['District IRN'].isin(public_1200_plus_ids))].copy()
other_public_df['Group'] = '<=1200 Hours'

combined = pd.concat([comparison_df, other_public_df])

# Visualize comparisons

# Enrollment size comparison

# New scatter plot showing relationship between enrollment and instructional hours
st.subheader("Instructional Hours vs Enrollment Size (Public Districts)")

# Ensure enrollment_df is loaded
overview_df = pd.read_excel("DISTRICT_HIGH_LEVEL_2324 (1).xlsx", sheet_name="DISTRICT_OVERVIEW")
enrollment_df = overview_df[['District IRN', 'Enrollment 2023-2024']].dropna()

public_all = pub_merge[pub_merge['District Type'] == 'Public']
public_enrollment = pd.merge(public_all, enrollment_df, on='District IRN', how='left')

fig_scatter, ax_scatter = plt.subplots()
sns.scatterplot(data=public_enrollment, x='Enrollment 2023-2024', y='Avg Instructional Hours', ax=ax_scatter)
ax_scatter.axhline(1200, color='red', linestyle='--', label='1200 Hour Threshold')
ax_scatter.set_title("Instructional Hours vs Enrollment Size (Public Districts)")
ax_scatter.set_xlabel("Enrollment (2023–24)")
ax_scatter.set_ylabel("Average Instructional Hours")
ax_scatter.legend()
ax_scatter.grid(True)
st.pyplot(fig_scatter)

overview_df = pd.read_excel("DISTRICT_HIGH_LEVEL_2324 (1).xlsx", sheet_name="DISTRICT_OVERVIEW")
enrollment_df = overview_df[['District IRN', 'Enrollment 2023-2024']].dropna()
combined = pd.merge(combined, enrollment_df, on='District IRN', how='left')

st.subheader("Enrollment Size by Instructional Hour Group")
fig_enroll, ax_enroll = plt.subplots()
sns.boxplot(data=combined, x='Group', y='Enrollment 2023-2024', ax=ax_enroll)
ax_enroll.set_title("Enrollment Size by Instructional Hour Group")
ax_enroll.set_ylabel("Enrollment (2023–24)")
st.pyplot(fig_enroll)
st.subheader("Comparison: High-Hour Public vs Other Public Districts")
for var in ['Chronic Absenteeism Rate', 'Average AGI', 'Expenditures', 'Teachers_FTE']:
    fig, ax = plt.subplots()
    sns.boxplot(data=combined, x='Group', y=var, ax=ax)
    ax.set_title(f"{var} by Instructional Hour Group")
    ax.set_ylabel(var)
    st.pyplot(fig)

st.subheader("Instructional Hours by Rural Status")

# Load rural typology data
rural_df = pd.read_excel("2013-School-District-Typology.xlsx")
rural_df = rural_df[rural_df['2013 Typology'].isin([6, 7])][['District Name']]
rural_df['Is Rural'] = 1

# Merge with instructional hours
typology_merge = data.merge(rural_df, on='District Name', how='left')
typology_merge['Is Rural'] = typology_merge['Is Rural'].fillna(0)

fig_rural, ax_rural = plt.subplots()
sns.boxplot(data=typology_merge, x='Is Rural', y='Avg Instructional Hours', ax=ax_rural)
ax_rural.set_xticklabels(['Non-Rural', 'Rural'])
ax_rural.set_title("Average Instructional Hours by Rural vs Non-Rural")
ax_rural.set_xlabel("District Type")
ax_rural.set_ylabel("Average Instructional Hours")
st.pyplot(fig_rural)


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



