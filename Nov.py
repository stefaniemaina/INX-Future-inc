import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split

#add title and headers
st.title("INX Future Inc. Employee Performance")

st.subheader("This is an employee performance predictor")

st.write("INX Future Inc is one of the leading data analytics and automation solutions provider with over 15 years of global business presence")
st.write("This model will help the company predict employee performance")

#load data
df= pd.read_excel("INX_Future_Inc_Employee_Performance .xlsx")

#show sample data
st.subheader("Sample Data")
st.dataframe(df.head())

# columns to update
columns_to_update = ('EmpEnvironmentSatisfaction' ,'EmpLastSalaryHikePercent' ,'YearsSinceLastPromotion' ,'ExperienceYearsInCurrentRole' ,'EmpDepartment' ,'EmpJobRole' ,'EmpWorkLifeBalance' ,'ExperienceYearsAtThisCompany' ,'YearsWithCurrManager')

valid_columns = df.columns.intersection(columns_to_update)


# Add two columns for data visualization
col1, col2 = st.columns(2)

with col1:
    st.header("Correlation Heatmap")
    f, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f) 

with col2:
    # Display a clustermap
    st.header("Clustermap")
    f, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", mask=mask, 
                cmap='coolwarm', vmin=-1, vmax=1) 
    st.pyplot(f) 

# Create our data features and target
X = df.drop(columns='PerformanceRating')
y = df['PerformanceRating']

# Columns to fit the model
columns = ['EmpEnvironmentSatisfaction' ,'EmpLastSalaryHikePercent' ,'YearsSinceLastPromotion' ,'ExperienceYearsInCurrentRole' ,'EmpDepartment' ,'EmpJobRole' ,'EmpWorkLifeBalance' ,'ExperienceYearsAtThisCompany' ,'YearsWithCurrManager']
X = X[columns]

#scale the data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_replaced = X_train.replace([np.inf, -np.inf], np.nan)
X_train_filled = X_train_replaced.fillna(X_train_replaced.mean())

X_test_replaced = X_test.replace([np.inf, -np.inf], np.nan)
X_test_filled = X_test_replaced.fillna(X_test_replaced.mean())
# getting user input
name = st.text_input('What is your name?').capitalize()
if name != "":
    st.write("Hello {} please fill below form by selecting from the slider on the sidebar".format(name))
else:
    st.write("Please enter your name")

# Get user input
def get_user_input():
    EmpEnvironmentSatisfaction = st.sidebar.slider("EmpEnvironmentSatisfaction", 1,4,3)
    EmpLastSalaryHikePercent = st.sidebar.slider("EmpLastSalaryHikePercent", 11,25,14)
    YearsSinceLastPromotion = st.sidebar.slider("YearsSinceLastPromotion", 0,15,1)
    ExperienceYearsInCurrentRole = st.sidebar.slider("ExperienceYearsInCurrentRole", 0,18,3)
    EmpDepartment = st.sidebar.slider("EmpDepartment", 0,5,4)
    EmpJobRole = st.sidebar.slider("EmpJobRole", 0,18,9)
    EmpWorkLifeBalance = st.sidebar.slider("EmpWorkLifeBalance", 1,4,3)
    ExperienceYearsAtThisCompany = st.sidebar.slider("ExperienceYearsAtThisCompany", 0,40,5)
    YearsWithCurrManager = st.sidebar.slider("YearsWithCurrManager", 0,17,3)

#Store in a dictionary
    user_data = {
        'EmpEnvironmentSatisfaction' : EmpEnvironmentSatisfaction,
        'EmpLastSalaryHikePercent' : EmpLastSalaryHikePercent,
        'YearsSinceLastPromotion' : YearsSinceLastPromotion,
        'ExperienceYearsInCurrentRole' : ExperienceYearsInCurrentRole,
        'EmpDepartment' : EmpDepartment,
        'EmpJobRole': EmpJobRole,
        'EmpWorkLifeBalance' : EmpWorkLifeBalance,
        'ExperienceYearsAtThisCompany' : ExperienceYearsAtThisCompany,
        'YearsWithCurrManager' : YearsWithCurrManager}
    
# Create features dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features
user_input = get_user_input()

#Displaying
st.subheader("Below is your input")
st.dataframe(user_input)

# Button for user to get results
bt = st.button("Get my results")

if bt:
    #create a gradient boosting classifier
    model = RandomForestClassifier(n_estimators = 100)
    model.fit(X_train, y_train)
    #get user input features
    prediction = model.predict(user_input)
    if prediction <3 :
        st.write("{}, your performance is rated 2".format(name))
    elif prediction >3 :
        st.write("{}, your performance is rated 4".format(name))
    else:
        st.write("{}, your performance is rated 3".format(name))

#Get model accuracy
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(bootstrap= False, max_depth= None, max_features= 'log2', min_samples_leaf= 1, min_samples_split= 2, n_estimators=  200)
model.fit(X_train_filled, y_train)
st.write("Model accuracy: ", round(metrics.accuracy_score(y_test, model.predict(X_test_filled)),2)*100)