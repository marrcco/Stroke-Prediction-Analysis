import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Reading the Data
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print(df.head())
print(df.info())

# # # TO DO
# CLEANING
# 1) FILL NULL VALUES IN BMI - DONE
# 2) CHECK ALL UNIQUE DATA IN GENDER, WORK TYPE, RESIDENCE  TYPE, SMOKING STATUS - DONE


# # # DATA CLEANSING

# Drop gender = other, because only there's only one
print(df["gender"].value_counts())
indexToDrop = df[df["gender"] == "Other"].index
df.drop(indexToDrop,inplace=True)


# There's so much Unknown data in smoking status
print(df["smoking_status"].value_counts())
print(df["hypertension"].value_counts())

df["smoking_status"] = df["smoking_status"].apply(lambda x: x.replace("formerly smoked","smokes"))


# Fill bmi nan data
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# # # LABEL ENCODING

le = LabelEncoder()

to_encode = ["gender","work_type","Residence_type","ever_married","smoking_status"]
def encode(colName):
    newName = colName + "_encoded"
    df[newName] = le.fit_transform(df[colName])
    return df

for x in to_encode:
    encode(x)


df.to_csv("healthcare_cleaned.csv",index=False)





