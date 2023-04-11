#Input the relevant libraries
import streamlit as st
import altair as alt
import nltk
import numpy as np
import pandas as pd
import base64
import nltk
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# Extract last N letters from the input word
# and that will act as our "feature"
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}

def assign_gender(name):
    return classifier.classify(extract_features(name,2))  

# Define the Streamlit app
def app():
    nltk.download('names')
    
    st.title("Gender Prediction from first names")      
    st.subheader("(c) 2023 Louie F. Cervantes, M.Eng.")
    
    st.subheader('The NLTK Names Package')
    st.write('The Natural Language Toolkit (NLTK) names package is a module in NLTK that provides a collection of datasets and functions for working with personal names. It includes datasets of names from various cultures and languages, as well as functions for generating random names, determining gender from a name, and identifying the most common prefixes and suffixes used in names.')
    
    with st.echo(code_location='below'):
         
        if st.button('Load Names from file'):
            # Create training data using labeled names available in NLTK
            male_list = [(name, 'male') for name in names.words('male.txt')]
            female_list = [(name, 'female') for name in names.words('female.txt')]
            data = (male_list + female_list)
            # Seed the random number generator
            random.seed(5)
            # Shuffle the data
            random.shuffle(data)  
            # Define the number of samples used for train and test
            num_train = int(0.8 * len(data))

            features = [(extract_features(n, 2), gender) for (n, gender) in data]
            train_data, test_data = features[:num_train], features[num_train:]
            classifier = NaiveBayesClassifier.train(train_data)

            # Compute the accuracy of the classifier
            accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
            st.write('Accuracy = ' + str(accuracy) + '%')
    
        st.subheader('Load the Alumni Data')
        st.write('The alumni data was encoded without a gender column.  We will use the machine learning approach to add the gender data to this dataset.')

        if st.button('Load the alumni data'):  
            df = pd.read_csv('2018-main.csv', header=0, sep = ",", encoding='latin')

            st.write('The data set before adding the gender')
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

            df['GENDER'] = df.apply(lambda row: assign_gender(row['FIRST NAME']), axis=1)

            st.write('The data set sfter adding the gender')
            st.dataframe(df.reset_index(drop=True), use_container_width=True)

    
# run the app
if __name__ == "__main__":
    app()
