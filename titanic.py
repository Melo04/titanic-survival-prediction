import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import time
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from streamlit_extras.let_it_rain import rain

st.set_page_config(
    page_title="Titanic Survival Prediction App",
    page_icon=":ship:",
)

progress_text = "Loading..."
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.05)
    my_bar.progress(percent_complete + 1, text=f"{percent_complete + 1}%  Loading...")
my_bar.empty()

st.write(
    """
    # Titanic Survival Prediction Web App :ship:
    Welcome Aboard :wave: :anchor:, this is a simple Titanic Survival Prediction web app made with Streamlit as well as Python and
    is capable of forecasting your likelihood of survival based on your input. Feel free to explore around :point_down:
    The data used to train the model were obtained from Kaggle. It contains
    demographics and passenger information from 891 of the 2224 passengers and crew
    on board the Titanic. You can check out the dataset [here](https://www.kaggle.com/c/titanic/data).
    """
)

image = Image.open('assets/titanic.jpg')
st.image(image, use_column_width=True)

class Toc:
    def __init__(self):
        self._items = []
        self._placeholder = None

    def header(self, text):
        self._markdown(text, "h2", " " * 2)

    def subheader(self, text):
        self._markdown(text, "h3", " " * 4)

    def placeholder(self, sidebar=False):
        self._placeholder = st.sidebar.empty() if sidebar else st.empty()

    def generate(self):
        if self._placeholder:
            self._placeholder.markdown("\n".join(self._items), unsafe_allow_html=True)
    
    def _markdown(self, text, level, space=""):
        key = re.sub('[^0-9a-zA-Z]+', '-', text).lower()

        style = 'font-size:1.5rem; font-weight:600; color:rgb(139, 255, 114); line-height 1.2;"'
        st.markdown(f"<p id='{key}' style={style}>{text}</p>", unsafe_allow_html=True)
        self._items.append(f"{space}* <a href='#{key}'>{text}</a>")


toc = Toc()
st.markdown('---')
st.title("Table of contents")
toc.placeholder()

st.markdown('---')
toc.header('Brief Introduction ✍️')
st.write("""
        The sinking of the Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew. 
        While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.
        This web app uses Random forest classifier model to predict the survival rate of the passengers.
        """)

st.markdown('---')
toc.header('Data Dictionary 📖')
st.write("""
        | Variable | Definition | Key |
        | --- | --- | --- |
        | Survival | Survival | 0 = No, 1 = Yes |
        | Pclass | Ticket class | 1 = 1st, 2 = 2nd, 3 = 3rd |
        | Sex | Sex of passenger | male/female |
        | Age | Age of passenger |  |
        | SibSp | # of siblings / spouses aboard the Titanic |  |
        | Parch | # of parents / children aboard the Titanic |  |
        | Embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |
        """)

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data = train_data.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis = 1)
test_data = test_data.drop(['PassengerId','Name','Ticket','Fare','Cabin'], axis = 1)

train_data = train_data.dropna(subset=['Embarked'], axis=0)
train_data['Age'].fillna(value = round(np.mean(train_data['Age'])), inplace = True)
test_data['Age'].fillna(value = round(np.mean(test_data['Age'])), inplace = True)

features = ["Pclass", "Age", "Sex", "SibSp", "Parch", "Embarked"]
X = pd.get_dummies(train_data[features])
Y = train_data["Survived"]
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, Y)
predictions = model.predict(X_test)

result_df = pd.DataFrame({
    'Pclass': test_data.Pclass,
    'Age': test_data.Age,
    'Sex': test_data.Sex,
    'SibSp': test_data.SibSp,
    'Parch': test_data.Parch,
    'Embarked': test_data.Embarked,
    'Survived': predictions
})
result_df['Pclass'] = result_df['Pclass'].map({1: '1st', 2: '2nd', 3: '3rd'})
result_df['Sex'] = result_df['Sex'].map({'male': 'Male', 'female': 'Female'})
result_df['Survived'] = result_df['Survived'].map({0: 'No', 1: 'Yes'})
if 'data/result.csv' in os.listdir():
    print('Model trained')
else:
    result_df.to_csv('data/result.csv', index=False)

@st.cache_data
def load_data(nrows):
    data = pd.read_csv('data/result.csv', nrows=nrows)
    return data

data = load_data(10000)
st.markdown('---')
toc.header('Result of Survival rate Among Titanic Passengers 🚢')
st.write('Below depicts the result of the survival rate among the Titanic passengers. The model is trained using Random Forest Classifier model and the data trained is obtained from Kaggle.')
st.write(data)

result_data = pd.read_csv('data/result.csv')
result_data['Survived'] = result_data['Survived'].map({'No': 0, 'Yes': 1})

st.markdown('---')
toc.header('Data Visualization 📊')
toc.subheader('Survival rate by passenger class')
survival_rates_by_class = pd.crosstab(result_data['Pclass'], result_data['Survived'])
survival_rates_by_class.columns = ['Does Not Survive', 'Survive']
st.bar_chart(survival_rates_by_class)

toc.subheader('Percentage of male vs female whom have survived')
women = train_data.loc[train_data.Sex == 'female']["Survived"]
men = train_data.loc[train_data.Sex == 'male']["Survived"]
women_survival_percentage = (women.sum() / len(women)) * 100
men_survival_percentage = (men.sum() / len(men)) * 100
st.bar_chart({'Male': men_survival_percentage, 'Female': women_survival_percentage})

toc.subheader('Number of passengers by age')
hist_values = np.histogram(
    train_data['Age'], bins=100, range=(0,100))[0]
st.line_chart(hist_values)

st.markdown('---')
toc.header('Predict Your survival probability on Titanic 💀')
form = st.form(key='my_form')
form.subheader('Enter your details and check your survival chances :crossed_fingers:')
toc.generate()

def user_input_features():
    passengerclass = form.radio('Choose passenger class', [1, 2, 3])
    gender = form.selectbox('Your gender', ('male', 'female'))
    age = form.slider('Your age', 0, 100, 20)
    sibsp = form.slider('Input siblings', 0, 10, 0)
    parch = form.slider('Input Parch (Parents/Children)', 0, 10, 0)
    embarked = form.radio('Choose embarkation point', ("S", "C", "Q"))

    gender_encoded = [1, 0] if gender == 'female' else [0, 1]

    embarked_encoded = [0, 0, 0] 
    if embarked == 'C':
        embarked_encoded = [1, 0, 0]
    elif embarked == 'Q':
        embarked_encoded = [0, 1, 0]
    else:
        embarked_encoded = [0, 0, 1]
        
    input_features = np.array([passengerclass, age, sibsp, parch] + gender_encoded + embarked_encoded).reshape(1, -1)
    return input_features

df = user_input_features()
predict = form.form_submit_button(label='Predict :crossed_fingers:')

if predict:
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader('Prediction Results :pray:')
    survival_chance = prediction_proba[0][1] * 100
    dead_chance = prediction_proba[0][0] * 100
    if prediction == 1:
        st.snow()
        st.success(f'Congrats! You have a higher chance in surviving :sunglasses: You will most likely end up like Rose :girl:')
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open('assets/meme3.jpg')
            st.image(image, width=300)
        with col2:
            image = Image.open('assets/survive.jpg')
            st.image(image, width=300)
    else:
        rain(
            emoji="💀",
            font_size=54,
            falling_speed=5,
            animation_length=0.5,
        )
        st.error(f'RIP :skull_and_crossbones:, You are most likely to be dead and ends up like Jack :man:')
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open('assets/dead.jpg')
            st.image(image, width=300)
        with col2:
            image = Image.open('assets/meme2.jpg')
            st.image(image, width=300)

    st.subheader('Survival Probability')
    st.success(f'Your surviving rate is {survival_chance:.2f}% :crossed_fingers:')
    st.error(f'Your death rate is {dead_chance:.2f}% :skull_and_crossbones:')