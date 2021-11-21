import keras
import pandas as pd
import numpy as np
import os
from pandas.core.algorithms import value_counts
import tensorflow as tf
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from itertools import islice
import random
from tensorflow import keras

from CSV import *

trainData_file_path = 'Data.csv'
testData_file_path = 'Evaluate.csv'


def file_size(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

def get_training_data(file_path):
    students_train_data = pd.read_csv(
        file_path
    )
    return students_train_data

def get_testing_data(file_path):
    student_test_data = pd.read_csv(
        file_path
    )
    return student_test_data


is_empty_train = file_size(trainData_file_path)
is_empty_test = file_size(testData_file_path)


if is_empty_train or is_empty_test:
    get_Data_file()
    print(get_training_data(trainData_file_path))

    get_Evaluate_file()
    print(get_testing_data(testData_file_path))

else:
    ask_for_update = input("Update files? [y]es [n]o:  ").upper()
    if ask_for_update == "Y":
        update_Data_file()
        print(get_training_data(trainData_file_path))
        update_Evaluate_file()
        print(get_testing_data(testData_file_path))
    if ask_for_update == "N":
        print(get_training_data(trainData_file_path))
        print(get_testing_data(testData_file_path))


# NORMALIZING TRAINING DATA
#---------------------------------------------------------------------
student_training_data = get_training_data(trainData_file_path).copy()

# vectorizing Gender
gender_allowed_values = ['M', 'F']
student_training_data.loc[~student_training_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

student_training_data['Gender'] = student_training_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

# vectorizing Language
languages_allowed_values = ['English', 'Spanish', 'Mandarin']
student_training_data.loc[~student_training_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

student_training_data['Description_HL'] = student_training_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'Mandarin', 'OTH'], value=[1, 2, 3, 0])

# vectorizing if in ELD
members_allowed_values = ['ELD', 'ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5'] #how many ELD groups are there?
student_training_data.loc[~student_training_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'
student_training_data['Group Memberships?'] = student_training_data['Group Memberships?'].replace(to_replace=['ELD','ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'OTH'], value=[1, 1, 1, 1, 1, 1, 0])


# cleaning data
student_training_data.drop(student_training_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)

print("\nModified Training Data: ")
print(student_training_data)




# BUILDING TRAINING MODEL
#---------------------------------------------------------------------
student_training_labels = student_training_data.pop('POD GROUP')

student_training_data = np.array(student_training_data)

def create_model():
    student_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='linear') 
    ])
    student_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer= tf.optimizers.Adam())

    return student_model

student_model = create_model()
student_model.summary()

                    

#student_training_data = np.asarray(student_training_data).astype('float32')
tf.convert_to_tensor(student_training_data)

student_model.fit(student_training_data, student_training_labels, epochs=50)


# NORMALIZING TESTING DATA
#---------------------------------------------------------------------
student_testing_data = get_testing_data(testData_file_path).copy()
student_testing_labels = student_testing_data.pop('POD GROUP')

# vectorizing Gender
student_testing_data.loc[~student_testing_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

student_testing_data['Gender'] = student_testing_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

# vectorizing Language
student_testing_data.loc[~student_testing_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

student_testing_data['Description_HL'] = student_testing_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'Mandarin', 'OTH'], value=[1, 2, 3, 0])

# vectorizing ELD
student_testing_data.loc[~student_testing_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'

student_testing_data['Group Memberships?'] = student_testing_data['Group Memberships?'].replace(to_replace=['ELD','ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'OTH'], value=[1,1, 1, 1, 1, 1, 0])

# cleaning data
student_testing_data.drop(student_testing_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)


print("\nModified Testing Data: ")
print(student_testing_data)


# TESTING MODEL
#---------------------------------------------------------------------
student_testing_data = np.array(student_testing_data)


print("\nTesting model: ")

student_model.evaluate(student_testing_data, student_testing_labels, verbose=1)

student_testing_labels = student_model.predict(student_testing_data)


print("\nData: \n{}".format(student_testing_data))
print('\nNot ELD: 0, ELD: 1') 


print("\nPod Group # Predictions (Not rounded): \n{}".format(abs(student_testing_labels)))
print('\nEnglish Group: 0, Spanish Group: 1, Mandarin Group: 2, Other Group: 3') 

# CREATING POD GROUPS
#---------------------------------------------------------------------

# adding data to 'POD GROUP' column
student_final_data = get_testing_data(testData_file_path).copy()
predictions = abs(student_testing_labels.round())
student_final_data['POD GROUP'] = predictions

print(student_final_data)

# adding students to dictionary
student_dictionary = student_final_data.to_dict(orient="index")


# adding students to their corresponding groups
english_group_name = []
spanish_group_name = []
mandarin_group_name = []
other_group_name = []

i = 0

while len(student_dictionary) != i:
    if student_dictionary[i]['POD GROUP'] == 0:
        english_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']}")

    if student_dictionary[i]['POD GROUP'] == 1:
        spanish_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']}")

    if student_dictionary[i]['POD GROUP'] == 2:
        mandarin_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']}")

    if student_dictionary[i]['POD GROUP'] == 3:
        other_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']}")
        
    i += 1




total_freshman_students = len(student_dictionary)
total_english_students = len(english_group_name)
total_spanish_students = len(spanish_group_name)
total_mandarin_students = len(mandarin_group_name)
total_other_students = len(other_group_name)


total_english_pod_groups = int(total_english_students/12) + 1
total_spanish_pod_groups = int(total_spanish_students/12) + 1
total_mandarin_pod_groups = int(total_mandarin_students/12) + 1
total_other_pod_groups = int(total_other_students/12) + 1

# Grouping pod groups by 12s
english_student_dictionary = {}
spanish_student_dictionary = {}
mandarin_student_dictionary = {}
other_student_dictionary = {}

random.shuffle(english_group_name)
random.shuffle(spanish_group_name)
random.shuffle(mandarin_group_name)
random.shuffle(other_group_name)



extra_english_students = []
extra_spanish_students = []
extra_mandarin_students = []
extra_other_students = []

def group_students(student_dictionary, group_name, total_students, group_number, extra_students):
    #if the group has less than 10 people put them in a seperate array -- different extra arrays for each language group
    i = 0

    while total_students != i:
        if i == 0 or i % 12 == 0:
            j = i + 12
            
            student_dictionary["Group {0}".format(group_number)] = group_name[i:j]

            if len(student_dictionary["Group {0}".format(group_number)]) < 10 and len(student_dictionary) > 1:
                extra_students.append(student_dictionary["Group {0}".format(group_number)])
                student_dictionary.pop("Group {0}".format(group_number))

            group_number += 1

        i += 1

    return student_dictionary


group_number = 1
group_students(english_student_dictionary, english_group_name, total_english_students, group_number,extra_english_students)

group_number += len(english_student_dictionary)
group_students(spanish_student_dictionary, spanish_group_name, total_spanish_students, group_number,extra_spanish_students)

group_number += len(spanish_student_dictionary)
group_students(mandarin_student_dictionary, mandarin_group_name, total_mandarin_students, group_number,extra_mandarin_students)

group_number += len(mandarin_student_dictionary)
group_students(other_student_dictionary, other_group_name, total_other_students, group_number,extra_other_students)


print("\nEnglish Groups: \n", english_student_dictionary)
print("\nSpanish Groups: \n", spanish_student_dictionary)
print("\nMandarin Groups: \n", mandarin_student_dictionary)
print("\nOther Groups: \n", other_student_dictionary)

print("\nEnglish Groups EXTRA: \n", extra_english_students)
print("\nSpanish Groups EXTRA: \n", extra_spanish_students)
print("\nMandarin Groups EXTRA: \n", extra_mandarin_students)
print("\nOther Groups EXTRA: \n", extra_other_students)

def add_extra_students(student_dictionary, extra_students, group_number, total_pods):
    #takes the extra people from each language group 
    #and adds them to the already existing groups (has to be in the same language group tho)
    #looping from the start until all extra people are gone
    i = 0
    starting_group_number = group_number
    if len(extra_students) > 0:
        count = len(extra_students[0])
        while count != 0:
            if (group_number - starting_group_number) == total_pods:
                group_number = starting_group_number

            student_dictionary["Group {0}".format(group_number)].append(extra_students[0][i])
            i += 1
            group_number += 1
            count -= 1
    

    
    return student_dictionary


group_number = 1
total_english_pod_groups = len(english_student_dictionary)
add_extra_students(english_student_dictionary, extra_english_students, group_number, total_english_pod_groups)

print("\nEnglish Groups: \n", english_student_dictionary)
print("\nEnglish Groups EXTRA: \n", extra_english_students)

group_number += len(english_student_dictionary)
total_spanish_pod_groups = len(spanish_student_dictionary)
add_extra_students(spanish_student_dictionary, extra_spanish_students, group_number, total_spanish_pod_groups)

print("\nSpanish Groups: \n", spanish_student_dictionary)
print("\nSpanish Groups EXTRA: \n", extra_spanish_students)

group_number += len(spanish_student_dictionary)
total_mandarin_pod_groups = len(mandarin_student_dictionary)
add_extra_students(mandarin_student_dictionary, extra_mandarin_students, group_number, total_mandarin_pod_groups)

print("\nMandarin Groups: \n", mandarin_student_dictionary)
print("\nMandarin Groups EXTRA: \n", extra_mandarin_students)

group_number += len(mandarin_student_dictionary)
total_other_pod_groups = len(other_student_dictionary)
add_extra_students(other_student_dictionary, extra_other_students, group_number, total_other_pod_groups)

print("\nOther Groups: \n", other_student_dictionary)
print("\nOther Groups EXTRA: \n", extra_other_students)


# FINALIZING POD GROUPS
#---------------------------------------------------------------------
all_pod_groups_dictionary = {}

all_pod_groups_dictionary.update(english_student_dictionary)
all_pod_groups_dictionary.update(spanish_student_dictionary)
all_pod_groups_dictionary.update(mandarin_student_dictionary)
all_pod_groups_dictionary.update(other_student_dictionary)

number_of_total_pod_groups = len(all_pod_groups_dictionary)

print(all_pod_groups_dictionary)


# converting dictionary to list
all_pod_groups_list = []

i = 1
while i != number_of_total_pod_groups:
    all_pod_groups_list.append(all_pod_groups_dictionary['Group {0}'.format(i)])
    i += 1


#print(all_pod_groups_list)

# SAVING THE MODEL (2 Methods)
#---------------------------------------------------------------------
# Create and train a new model instance.
#model = create_model()
#model.fit(student_training_data, student_training_labels, epochs=250)

# Save the entire model as a SavedModel.
#model.save('saved_student_model/my_model')

#---------------------------------------------------------------------
# Create and train a new model instance.
#model_h5 = create_model()
#model_h5.fit(student_training_data, student_training_labels, epochs=250)

# Save the entire model to a HDF5 file.
#model_h5.save('my_model.h5')






# EXTRA MODELS
#---------------------------------------------------------------------
#Decision Tree Model

#DT_model = DecisionTreeRegressor(max_depth=5).fit(student_training_data,student_training_labels)
#DT_predict = DT_model.predict(student_testing_data)
#DT_score = (student_testing_data, student_testing_labels)

#K-nearest-neighbor Model

#KNN_model = KNeighborsRegressor(n_neighbors=5).fit(student_testing_data,student_testing_labels)
#KNN_predict = KNN_model.predict(student_testing_data) 
#score = KNN_model.score(student_testing_data, student_testing_labels)










    




