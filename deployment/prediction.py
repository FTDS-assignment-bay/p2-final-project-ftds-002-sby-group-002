import pandas as pd
import streamlit as st
import pickle
from PIL import Image
import json 
import numpy as np 
from sklearn.decomposition import PCA

# Load for prediction
with open('model.pkl','rb') as file_1:
    model = pickle.load(file_1)

# Load for Clustering
with open('list_num_columns.txt', 'r') as file_1: # To load list column
    list_num = json.load(file_1)
with open('list_cat_columns.txt', 'r') as file_2: # To load list column
    list_cat = json.load(file_2)
with open('model_scaler.pkl', 'rb') as file_3: # To load scaling numeric 
    scaler = pickle.load(file_3)
with open('KPrototypes.pkl', 'rb') as file_4: # To load model kmeans
    kpmodel = pickle.load (file_4)
with open('pca.pkl', 'rb') as file_5: # To load model pca
    pca = pickle.load (file_5)
def run():
    # Create form
    image = Image.open('Logo_Abank.png')
    st.image(image,caption="Abank")
    with st.form('form_client') :
       age = st.number_input('Age', 
                            min_value = 1, max_value = 87, 
                            value=1,
                            step = 1,
                            help='Input age of client') 
       job = st.radio('Job',('unemployed', 'services', 'management', 
                             'blue-collar','self-employed', 'technician', 
                             'entrepreneur', 'admin.', 'student','housemaid',
                               'retired', 'unknown'),index=0,help='Input client job')
       marital = st.radio('Marital Status',('married', 'single', 'divorced'),
                          index=0,help='Input marital status of client ')
       education = st.radio('Education Status',('primary', 'secondary', 'tertiary',
                                                 'unknown'),index=0,
                                                 help='Input education status of client ')
       credit = st.radio('Has Credit',('no','yes'),index=0,
                         help='Are client has credit ?')
       st.write('---')

       balance = st.slider('Balance',-3313,71188,0, help='Input client balance')
       st.write('---')

       housing = st.radio('Housing Loan',('no','yes'),index=0,
                          help='Are the client has housing loan?')
       loan = st.radio('Client Loan',('no','yes'),index=0,
                          help='Are the client has loan?')
       contact = st.radio('Client type of Contact',('cellular', 'unknown', 'telephone'),
                          index=0, help='What type of client contact?')
       st.write('---')

       day = st.slider('Day',1,31,0, help='Input the day')
       
       st.write('---')

       month = st.radio('Month',('october', 'may', 'april', 'juny', 'february', 'august', 'january',
                                 'july', 'november', 'september', 'march', 
                                 'december'),index=0,help='Input the month')
       
       st.write('---')

       duration = st.slider('Duration',4,3025,0, 
                                   help='Duration of client Call')
       campaign =  st.slider('Campaign',1,50,0, 
                                   help='Campaign that Bank ever did')
       days_pass =  st.slider('Days Passed',-1,871,0, 
                                   help='Days passed after client last contacted after campaign')
       previous =  st.slider('Previous client contact',-1,871,0, 
                                   help='Number of contacted client before campaign')
       
       st.write('---')

       outcome_pass = st.radio('Outcame from campaign',('unknown', 'failure', 'other', 'success'),
                          index=0, help='Are the outcame from passed success?') 
       
       submitted = st.form_submit_button('Predict')
       
       new_duration = []

    def get_duration(data):
        if data < 300:
            new_duration.append("Long duration")
        else:
            new_duration.append("Short duration")
    generations = []

    def get_generation(data):
        if 10 < data <= 25:
            generations.append("Gen Z")
        elif 26 < data <= 43:
            generations.append("Millennials")
        elif 44 < data <= 58:
            generations.append("Gen X")
        elif 59 < data <= 78:
            generations.append("Baby Boomers")
        else:
            generations.append("Silent Generation")

   

       # Data inference
    data_inf = {
           'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'has_credit': credit,
            'balance': balance,
            'housing_loan': housing,
            'loan': loan,
            'contact': contact,
            'day': day, 
            'month': month,
            'duration': duration,
            'campaign': campaign,
            'days_passed': days_pass,
            'previous': previous,
            'outcome_passed': outcome_pass,
            }
    
    data_inf = pd.DataFrame([data_inf])
    for duration_value in data_inf["duration"]:
        get_duration(duration_value)
    data_inf["duration_category"] = new_duration
    for age_value in data_inf["age"]:
        get_generation(age_value)

    data_inf["generation"] = generations
    st.dataframe(data_inf)

    if submitted:
        
        # Predict Classification
        y_inf_pred = model.predict(data_inf)
        data_inf['subscribed'] = y_inf_pred
        # if y_inf_pred == 0:
        #     data_inf_num_col = data_inf[list_num]
        #     data_inf_cat_col = data_inf[list_cat]
        #     list_num_s = scaler.transform(data_inf_num_col)
        #     # list_num_pca = pca.transform(list_num_s)
        #     df_final = np.concatenate([list_num_s, data_inf_cat_col], axis=1)
        #     df_final = pd.DataFrame(df_final, columns=["age", "balance", "day", 
        #                                                "duration", "campaign", "days_passed",
        #                                                 "previous"] + list_cat)
        #     df_final = df_final.infer_objects()

        #     # Get the position of categorical columns

        #     index_cat_columns = [df_final.columns.get_loc(col) for col in list_cat]
        #     # Predict Cluster
        #     y_cluster = kpmodel.predict(df_final, categorical=index_cat_columns)
        #     data_inf['cluster'] = y_cluster
        #     st.write('# Cluster :', y_cluster)
        #     # Preduct Clustering
        #     st.write('# Subscribed : ', str(y_inf_pred)) 
        # else:
        st.write('# Subscribed : ', str(y_inf_pred))
# if y_inf_pred == 1 :
#     y_cluster = model.predict(data_inf)
#     str.write('### Client termasuk kedalam kelompok',y_cluster)
if __name__ == '__main__':
    run()