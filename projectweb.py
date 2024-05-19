#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import pickle
import pandas as pd

df = pd.read_csv('TTest.csv')
def classify_satisfaction(prediction):
    return 'Not Satisfied' if prediction == 1 else 'Satisfied'

def main():
    st.title("Airline Passenger Satisfaction Prediction")

    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white; text-align:center;">Predict Passenger Satisfaction</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)   

    # Load the trained SVM model
    log_model_saved = pickle.load(open('log_model_re.pkl', 'rb'))

    # Collect input features from the user
    Gender = st.slider('Gender (M=1,F=0)', 1, 0, key='gender_slider')
    Coustomer_type = st.slider('Customer Type  (Loyal=1, disLoyal=0)', 1, 0, key='customer_type_slider')
    age = st.number_input('Age', min_value=0, max_value=150, value=30, key='age_input')
    Class = st.slider('Class (Business=1,Eco Plus=2, Eco=3)', 1, 2,3, key='class_slider')  
    Inflight_wifi_service = st.slider('Inflight wifi service (strongly agree=5, disagree=1)', 1, 5, key='wifi_slider') 
    Ease_of_Online_booking = st.slider('Ease of Online booking (strongly agree=5, disagree=1)', 1, 5, key='booking_slider')
    food_and_drink = st.slider('Food and drink (strongly agree=5, disagree=1)', 1, 5, key='food_slider')
    Seat_comfort = st.slider('Seat comfort (strongly agree=5, disagree=1)', 1, 5, key='seat_slider')
    On_board_service = st.slider('On-board service (strongly agree=5, disagree=1)', 1, 5, key='onboard_slider') 
    Baggage_handling = st.slider('Baggage handling (strongly agree=5, disagree=1)', 1, 5, key='baggage_slider')  
    Checkin_service = st.slider('Checkin service (strongly agree=5, disagree=1)', 1, 5, key='checkin_slider') 
    Inflight_service_1 = st.slider('Inflight service 1 (strongly agree=5, disagree=1)', 1, 5, key='inflight_service1_slider')
    Inflight_service_2 = st.slider('Inflight service 2 (strongly agree=5, disagree=1)', 1, 5, key='inflight_service2_slider')
    Departure_Delay_in_Minutes = st.slider('Departure Delay in Minutes  (strongly agree=5, disagree=1)', 1, 5, key='departure_slider')
   

    # Prepare input data for prediction
    user_input = [[ Gender, Coustomer_type,age ,Class,Inflight_wifi_service
                   ,Ease_of_Online_booking,food_and_drink,Seat_comfort,On_board_service,Baggage_handling,Baggage_handling
                   ,Checkin_service,Inflight_service_1,Departure_Delay_in_Minutes,Inflight_service_2]]

    if st.button('Predict Satisfaction'):
        # Predict passenger satisfaction using the loaded SVM model
        prediction = log_model_saved.predict(user_input)
        satisfaction_result = classify_satisfaction(prediction[0])

        # Display the predicted satisfaction
        st.success(f"The passenger is predicted to be: {satisfaction_result}")

if __name__ == '__main__':
    main()






# In[ ]: