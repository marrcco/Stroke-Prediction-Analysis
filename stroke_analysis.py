import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


df = pd.read_csv("healthcare_cleaned.csv")



# # # VISUALIZATIONS
# VISUALIZATIONS
# 1) MARRIAGE AFFECT ON STROKE - DONE
# 2) GENDER AFFECT ON STROKE - DONE
# 3) AGE AFFECT ON STROKE - DONE
# 4) AGE AND GENDER AFFECT ON STROKE - DONE
# 5) HEART DISEASE AFFECT ON STROKE - DONE
# 6) BMI + AGE AFFECT ON STROKE - DONE
# 7) AVG GLUCOSE AFFECT ON STROKE - DONE
# 8) SMOKE AFFECT ON STROKE - DONE
# 9) HYPERTENSION AFFECT ON STROKE - DONE
# 10) WORK TYPE AFFECT ON STROKE


# Marriage and Age affect on Stroke
marriageStrokeAffect = sns.countplot(x="stroke",hue="ever_married",data=df,palette="husl")
marriageStrokeAffect.set_title("Marriage Affect on Stroke")
plt.show()


# Closer look for Strokes and Marriage
onlyStroke = df[df["stroke"] == 1]
marriageStrokeAffectOnlyStroke = sns.countplot(x="stroke",hue="ever_married",data=onlyStroke,palette="husl")
marriageStrokeAffectOnlyStroke.set_title("Marriage Affect On Stroke")
plt.show()


# Age Affect on Stroke
f, ax = plt.subplots(figsize=(18, 7))

ageStroke = sns.countplot(x="age",hue="stroke",data=onlyStroke,palette="husl")
ageStroke.set_title("Age Affect on Stroke")
plt.show()

# Gender Affect on Stroke
genderStroke = sns.countplot(x="stroke",hue="gender",data=onlyStroke,palette="Set2")
genderStroke.set_title("Gender Affect on Stroke")
plt.show()

# Gender and Age Affect on Stroke
ageGenderStrokeFig = px.bar(onlyStroke,x="age",y="stroke",color="gender",title="Age and Gender Affect on Stroke")
ageGenderStrokeFig.show()

# Heart Disease affect on Stroke
heartDiseaseStroke = px.bar(onlyStroke,x="heart_disease",y="stroke",title="Heart Disease affect on Stroke")
heartDiseaseStroke.show()

# Heart Disease and Gender affect on Stroke
heartDiseaseStrokeGender = px.bar(onlyStroke,x="heart_disease",y="stroke",color="gender",title="Heart Disease affect on Stroke")
heartDiseaseStrokeGender.show()


# BMI and Age AFfect on Stroke
df["stroke_str"] = df["stroke"].apply(str)
bmiAgeStroke = px.scatter(df,x="age",y="bmi",color="stroke_str",color_discrete_sequence=px.colors.qualitative.Set3,
                          title="Age and BMI Affect on Stroke")
bmiAgeStroke.show()

# Glucose Affect on Stroke
glucoseAgeStroke = px.scatter(df,x="age",y="avg_glucose_level",color="stroke",
                              color_discrete_sequence=px.colors.qualitative.Set3,
                              title="Glucose Level and Age Affect on Stroke")
glucoseAgeStroke.show()


# Smoke Affect on Stroke
smokeStroke = px.bar(onlyStroke,x="smoking_status",y="stroke",title="Smoke Affect on Stroke")
smokeStroke.show()

# Hypertension Affect on Stroke
onlyHyperTension = df[df["hypertension"] == 1]
hypertensionStroke = px.bar(onlyHyperTension,x="stroke",y="hypertension",title="Hypertension Affect on Stroke")
hypertensionStroke.show()

hypertensionStroke = sns.countplot(x="hypertension",hue="stroke",data=onlyHyperTension)
hypertensionStroke.set_title("Hypertension Affect on Stroke")
plt.show()

# Work Type Affect on Stroke
workStroke = px.bar(df,x="work_type",y="stroke",color="gender",title="Work Type Affect on Stroke")
workStroke.show()

# Residence Type Affect on Stroke
residenceStroke = sns.countplot(x="Residence_type",data=df,hue="stroke")
plt.show()

# Residence Type Affect on Stroke
residenceStroke1 = sns.countplot(x="Residence_type",data=onlyStroke,hue="stroke")
plt.show()

# # # CORRELATION
corr_df = df.corr()
f, ax = plt.subplots(figsize=(10, 8))

corr_vis = sns.heatmap(corr_df,cmap="YlGnBu")
plt.show()

