#  https://drive.google.com/drive/folders/15FZGK4aoDx6NQaA6wv2Fo39HqFVO3Hev
## a. t test 
import pandas as pd
from scipy.stats import ttest_ind
df=pd.read_csv(r'C:\Users\user\Downloads\StudentsPerformance.csv')
df
# seperate data for female students
female_scores=df[df['gender']=='female']['math score']
female_scores
# seperate data for male students 
male_scores=df[df['gender']=='male']['math score']
male_scores
# perform 2 sample indepenent t test
t_statistic,p_value=ttest_ind(male_scores,female_scores)

print(f'T-Statistic:{t_statistic}')
print(f'p_value: {p_value}')
# interpreting result
alpha=0.5
if p_value<alpha:
    print("There is no significant difference between the maths scores of male and female students")
else: 
    print("There is a significant difference ebtween the math scores of male and female students")


## b. Extracting math scores of each ethinicities
from scipy_stats import f_oneway

df=pd.read_csv(r'C:\Users\user\Downloads\StudentsPerformance.csv')

ethnicity_groups = df['ethnicity'].unique()
ethnicity_groups
# 
ethnicity_data = {ethnicity: df[df['ethnicity'] == ethnicity]['math score'] for ethnicity in ethnicity_groups}
ethnicity_data
# Perform ANOVA
f_statistic, p_value_anova = f_oneway(*ethnicity_data.values())
print(f'f-statistic: {f_statistic}')
print(f'p-value (ANOVA): {p_value_anova}')
# interpret result
if p_value_anova<alpha:
    print("There is no significant difference between scores among different enthicities")
else:
    print("There is a significant difference among different enthicities")
