import pandas as pd
import numpy as np

#LOAD THE DATASET FIRST
#df = pd.read_csv(------DATASET PATH------)

# EXAMPLE DATASET 
df = pd.read_csv(r"C:\Users\rgour\Downloads\data_science_job.csv")

df['years_since_start'] = 2024 - df['work_year']  # Assume 2024 as the current year

df['salary_ratio'] = df['salary_in_usd'] / df['salary']  # Ratio of USD salary to original salary
df['salary_log'] = np.log1p(df['salary_in_usd'])  # Log-transform to reduce skewness in salary

df = pd.get_dummies(df, columns=['experience_level', 'employment_type', 'work_setting', 'company_size'])

df['avg_salary_job_category'] = df.groupby('job_category')['salary_in_usd'].transform('mean')
df['avg_salary_job_title'] = df.groupby('job_title')['salary_in_usd'].transform('mean')

df['same_location'] = (df['company_location'] == df['employee_residence']).astype(int)  # 1 if same location, else 0

# Step 6: Drop redundant columns if not needed for modeling (optional)
# e.g., df_fe.drop(columns=['work_year', 'salary_currency', 'salary'], inplace=True)

print(df.head())
