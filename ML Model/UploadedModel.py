import tensorflow as tf
import pandas as pd
import numpy as np

testData_file_path = 'Evaluate.csv'


def get_testing_data(file_path):
    student_test_data = pd.read_csv(
        file_path
    )
    return student_test_data


# NORMALIZING TESTING DATA
#---------------------------------------------------------------------
student_testing_data = get_testing_data(testData_file_path).copy()
student_testing_labels = student_testing_data.pop('POD GROUP')

# vectorizing Gender
gender_allowed_values = ['M', 'F']
student_testing_data.loc[~student_testing_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'
student_testing_data['Gender'] = student_testing_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

# vectorizing Language
languages_allowed_values = ['English', 'Spanish', 'Mandarin']
student_testing_data.loc[~student_testing_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'
student_testing_data['Description_HL'] = student_testing_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'Mandarin', 'OTH'], value=[1, 2, 3, 0])

# vectorizing ELD
members_allowed_values = ['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5'] #how many ELD groups are there?
student_testing_data.loc[~student_testing_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'
student_testing_data['Group Memberships?'] = student_testing_data['Group Memberships?'].replace(to_replace=['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'OTH'], value=[1, 1, 1, 1, 1, 0])

# cleaning data
student_testing_data.drop(student_testing_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)


print("\nModified Testing Data: ")
print(student_testing_data)


# TESTING MODEL FROM THE UPLOADED MODEL (.h5 format)
#---------------------------------------------------------------------
student_testing_data = np.array(student_testing_data)

# Recreate the exact same model, including its weights and the optimizer
new_model_h5 = tf.keras.models.load_model('saved_student_model/my_model.h5')
loss_h5 = new_model_h5.evaluate(student_testing_data, student_testing_labels, verbose=1)

student_testing_labels = new_model_h5.predict(student_testing_data)
print('Predictions: \n', student_testing_labels)


# OTHER MODEL (.pb format)
#---------------------------------------------------------------------
#new_model = tf.keras.models.load_model('saved_student_model/my_model')
#new_model.summary()

#loss = new_model.evaluate(student_testing_data, student_testing_labels, verbose=1)

#student_testing_labels = new_model.predict(student_testing_data)
#print('Predictions: \n',student_testing_labels)


# PRINTING DATA AND ARCHITECTURE
#---------------------------------------------------------------------
student_final_data = get_testing_data(testData_file_path).copy()
predictions = abs(student_testing_labels.round())
student_final_data['POD GROUP'] = predictions

new_model_h5.summary()

print(student_final_data)