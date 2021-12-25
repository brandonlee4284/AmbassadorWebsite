from math import sqrt
from django.db.models.fields import GenericIPAddressField
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
import itertools
import threading
from threading import Thread

from CSV import *

trainData_file_path = 'Data.csv'
testData_file_path = 'Evaluate.csv'


class CreatePods:

    # NORMALIZING TRAINING DATA
    #---------------------------------------------------------------------
    file_path_train = trainData_file_path

    student_train_data = pd.read_csv(file_path_train)

    student_training_data = student_train_data.copy()

    # vectorizing Gender
    gender_allowed_values = ['M', 'F']
    student_training_data.loc[~student_training_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

    student_training_data['Gender'] = student_training_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

    # vectorizing Language
    languages_allowed_values = ['English', 'Spanish']
    student_training_data.loc[~student_training_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

    student_training_data['Description_HL'] = student_training_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'OTH'], value=[1, 2, 0])

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
    file_path = testData_file_path

    student_test_data = pd.read_csv(file_path)

    student_testing_data = student_test_data.copy()

    student_testing_labels = student_testing_data.pop('POD GROUP')

    # vectorizing Gender
    student_testing_data.loc[~student_testing_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

    student_testing_data['Gender'] = student_testing_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

    # vectorizing Language
    student_testing_data.loc[~student_testing_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

    student_testing_data['Description_HL'] = student_testing_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'OTH'], value=[1, 2, 0])

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
    student_final_data = student_test_data.copy()
    predictions = abs(student_testing_labels.round())
    student_final_data['POD GROUP'] = predictions

    print(student_final_data)

    # adding students to dictionary
    student_dictionary = student_final_data.to_dict(orient="index")

    random.shuffle(student_dictionary)

    # adding students to their corresponding groups
    english_group_dict = dict()
    spanish_group_dict = dict()
    other_group_dict = dict()

    # CREATING LANGUAGE GROUPS
    #---------------------------------------------------------------------

    def make_language_groups(student_dictionary, language_group_dict, pod_number, name):
        i = 0
        student_number = 1
        while len(student_dictionary) != i:
            if student_dictionary[i]['POD GROUP'] == pod_number:
                language_group_dict[name + " Student {0}".format(student_number)] = student_dictionary[i]
                student_number+=1
            
            
            i+=1
        
        return language_group_dict


    english_group_dict = make_language_groups(student_dictionary=student_dictionary, language_group_dict=english_group_dict,pod_number=0, name='English')
    spanish_group_dict = make_language_groups(student_dictionary=student_dictionary, language_group_dict=spanish_group_dict,pod_number=1, name='Spanish')
    other_group_dict = make_language_groups(student_dictionary=student_dictionary, language_group_dict=other_group_dict,pod_number=3, name='Other')

    #print(english_group_dict)
    #print("\n")
    #print(spanish_group_dict)
    #print("\n")
    #print(other_group_dict)

    english_pod_dictionary = dict()
    spanish_pod_dictionary = dict()
    other_pod_dictionary = dict()

    extra_english_students_dict = dict()
    extra_spanish_students_dict = dict()
    extra_other_students_dict = dict()

    total_english_students = len(english_group_dict)
    total_spanish_students = len(spanish_group_dict)
    total_other_students = len(other_group_dict)
    #total_students = total_english_students + total_spanish_students + total_other_students
    #print(total_students)

    #print(total_spanish_students + total_english_students + total_other_students)

    # ORGANIZING PODS
    #---------------------------------------------------------------------

    def make_pods(language_pod_dictionary_new, language_group_dict_old, total_students, group_number, extra_students_dict, name):
        #if the group has less than 8 people put them in a seperate array -- different extra arrays for each language group
        i = 0
        extra_student_number = 1
        counter = 0
        #extra_students_dict["Extra Students"] = {}
        while total_students != i:
            #assign multiple students to one key
            j = i + 12        

            if i % 12 == 0:
                language_pod_dictionary_new["Pod {0}".format(group_number)] = dict(itertools.islice(language_group_dict_old.items(), i, j))
                
                if len(language_pod_dictionary_new["Pod {0}".format(group_number)]) < 8 and len(language_pod_dictionary_new) > 1:
                    k = 0
                    while k != len(language_pod_dictionary_new["Pod {0}".format(group_number)]):
                        extra_students_dict["Extra Students {0}".format(extra_student_number)] = dict(itertools.islice(language_pod_dictionary_new["Pod {0}".format(group_number)].items(), counter, counter+1)) 
                        extra_student_number+=1
                        counter += 1
                        k += 1

                    
                ##extra_students_dict["Extra Students"][name+" Extra Student {0}".format(1)] = extra_students_dict[name+" Extra Student {0}".format(extra_student_number)]

                    language_pod_dictionary_new.pop("Pod {0}".format(group_number))

                group_number += 1
                

            
                
            i += 1
            
        return language_pod_dictionary_new

    group_number = 1
    make_pods(english_pod_dictionary, english_group_dict, total_english_students, group_number, extra_english_students_dict, "English")

    group_number += len(english_pod_dictionary)
    make_pods(spanish_pod_dictionary, spanish_group_dict, total_spanish_students, group_number, extra_spanish_students_dict, "Spanish")

    group_number += len(spanish_pod_dictionary)
    make_pods(other_pod_dictionary, other_group_dict, total_other_students, group_number, extra_other_students_dict, "Other")

            
    #print("\nEnglish Groups: \n", english_pod_dictionary)
    #print("\nEnglish Extra Students: \n", extra_english_students_dict)

    #print("\nSpanish Groups: \n", spanish_pod_dictionary)
    #print("\nSpanish Extra Students: \n", extra_spanish_students_dict)

    #print("\nOther Groups: \n", other_pod_dictionary)
    #print("\nOther Extra Students: \n", extra_other_students_dict)

    newly_add_pods_dict = dict()

    # REGROUPING PODS TO ADD EXTRA STUDENTS
    #---------------------------------------------------------------------

    def add_extra_students(self,language_pod_dictionary, group_number, extra_students_dict, newly_add_pods_dict, name):
        #takes the extra people from each language group 
        #and adds them to the already existing groups (has to be in the same language group)
        #looping from the start until all extra people are gone
        total_pods = len(language_pod_dictionary)
        count = len(extra_students_dict)
        starting_group_number = group_number
        i = 0
        extra_students_counter = 1
        

        if len(extra_students_dict) > 0:

            if (len(language_pod_dictionary) >= 7) or \
                (len(language_pod_dictionary) == 1 and len(extra_students_dict) <= 1) or \
                (len(language_pod_dictionary) == 2 and len(extra_students_dict) <= 2) or \
                (len(language_pod_dictionary) == 3 and len(extra_students_dict) <= 3) or \
                (len(language_pod_dictionary) == 4 and len(extra_students_dict) <= 4) or \
                (len(language_pod_dictionary) == 5 and len(extra_students_dict) <= 5) or \
                (len(language_pod_dictionary) == 6 and len(extra_students_dict) <= 6): 
                while count != 0:
                    if (group_number - starting_group_number) == total_pods:
                        group_number = starting_group_number

                    #language_pod_dictionary["Pod {0}".format(group_number)].update(extra_students_dict["Extra Students"])
                    #extra_students_dict.pop("Extra Students")
                    #for index, key in enumerate(extra_students_dict["Extra Students"]):
                    language_pod_dictionary["Pod {0}".format(group_number)].update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        #extra_students_dict["Extra Students"].pop(str(key))
                        
                    group_number += 1
                    extra_students_counter += 1
                    count -= 1
                    i+=1


            # EDGE CASES (CONDITIONALLY)
            # 1
            #------------------------------------
            if len(language_pod_dictionary) == 1: 

                if len(extra_students_dict) == 2:
                    # 2 groups: (7,7)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]))/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 3:
                    # 2 groups: (7,8)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]) + 1)/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 4:
                    # 2 groups: (8,8)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]))/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 5:
                    # 2 groups: (8,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]) + 1)/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                    
                if len(extra_students_dict) == 6:
                    # 2 groups: (9,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]))/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 7:
                    # 2 groups: (10,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])

                        group_number += 1
                        extra_students_counter += 1
                        count -= 1
                    
                    half_way_index = int((len(language_pod_dictionary["Pod {0}".format(group_number-1)]) + 1)/2)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, half_way_index))

                    #for x in range(half_way_index):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))
                    
                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < half_way_index:
                            keys_to_remove_1.append(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

            # 2
            #------------------------------------
            if len(language_pod_dictionary) == 2: 
        
                if len(extra_students_dict) == 3:
                    # 3 groups: (9,9,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 5 # change
                    index_2 = 4 # change
                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))
                    

                    
                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)

                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]
                        


                if len(extra_students_dict) == 4:
                    # 3 groups: (10,9,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 4 # change
                    index_2 = 5 # change

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_2)))
                    
                    #next_start = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]


                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)

                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 5:
                    # 3 groups: (10,10,9)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 5 # change
                    index_2 = 4 # change
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))
                    
                    #next_start = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)

                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                if len(extra_students_dict) == 6:
                    # 3 groups: (10,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 5 # change
                    index_2 = 5 # change

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_2)))
                    
                    #next_start = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]


                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)

                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                if len(extra_students_dict) == 7:
                    # 3 groups: (11,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 5 # change
                    index_2 = 5 # change
                
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))
                    
                    #next_start = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)

                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)

                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]

            # 3
            #------------------------------------
        
            if len(language_pod_dictionary) == 3: 

                if len(extra_students_dict) == 4:
                    # 4 groups: (10,10,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 4 # change
                    index_2 = 3 # change
                    index_3 = 3 # change

                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_3)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]

                if len(extra_students_dict) == 5:
                    # 4 groups: (11,10,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 4 # change
                    index_3 = 3 # change

                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_3)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]

                if len(extra_students_dict) == 6:
                    # 4 groups: (11,11,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 3 # change
                    index_3 = 4 # change

                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-3)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_3)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-3)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-3)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_2+1))
                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-3)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-3)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                if len(extra_students_dict) == 7:
                    # 4 groups: (11,11,11,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 4 # change
                    index_2 = 3 # change
                    index_3 = 3 # change

                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_3)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]

            # 4
            #------------------------------------
            

            if len(language_pod_dictionary) == 4: 

                if len(extra_students_dict) == 5:
                    # 5 groups: (11,11,11,10,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 2 # change
                    index_3 = 2 # change
                    index_4 = 3 # change

                    #print(group_number)
                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+2)].items(), 0, index_4)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+2)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number+2)][key]


                if len(extra_students_dict) == 6:
                    # 5 groups: (11,11,11,11,10)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 3 # change
                    index_3 = 2 # change
                    index_4 = 2 # change
                    #print(group_number)

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_4)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_3+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]


                if len(extra_students_dict) == 7:
                    # 5 groups: (11,11,11,11,11)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 3 # change
                    index_3 = 3 # change
                    index_4 = 2 # change
                    #print(group_number)

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-3)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_4)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-3)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-3)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_3+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-3)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-3)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


            # 5
            #------------------------------------
            
            if len(language_pod_dictionary) == 5: 
            
                if len(extra_students_dict) == 6:
                    # 6 groups: (11,11,11,11,11,11)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 3 # change
                    index_2 = 2 # change
                    index_3 = 2 # change
                    index_4 = 2 # change
                    index_5 = 2 # change
                    #print(group_number) #65

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+2)].items(), 0, index_4)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+3)].items(), 0, index_5)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                    #next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+2)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                    #for x in range(index_5):
                        #language_pod_dictionary["Pod {0}".format(group_number+3)].pop(name + " Student {0}".format(x+next_start_4+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+2)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number+2)][key]

                    
                    keys_to_remove_5 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+3)]): 
                        if index < index_5:
                            keys_to_remove_5.append(key)
                            #print(key)
                    for key in keys_to_remove_5:
                        del language_pod_dictionary["Pod {0}".format(group_number+3)][key]

                if len(extra_students_dict) == 7:
                    # 6 groups: (12,11,11,11,11,11)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 2 # change
                    index_2 = 3 # change
                    index_3 = 2 # change
                    index_4 = 2 # change
                    index_5 = 2 # change
                    print(group_number) #66

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-2)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_4)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+2)].items(), 0, index_5)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_3+1))

                    #for x in range(index_5):
                        #language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_4+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-2)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-2)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]

                    
                    keys_to_remove_5 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+2)]): 
                        if index < index_5:
                            keys_to_remove_5.append(key)
                            #print(key)
                    for key in keys_to_remove_5:
                        del language_pod_dictionary["Pod {0}".format(group_number+2)][key]

            # 6
            #------------------------------------

            if len(language_pod_dictionary) == 6: 
            
                if len(extra_students_dict) == 7:
                    # 7 groups: (12,12,11,11,11,11,11)
                    new_group_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + len(newly_add_pods_dict) + 1
                    while count != 0:
                        if (group_number - starting_group_number) == total_pods:
                            group_number = starting_group_number

                        language_pod_dictionary["Pod {0}".format(group_number)] \
                        .update(extra_students_dict["Extra Students {0}".format(extra_students_counter)])
                        
                        group_number += 1
                        extra_students_counter += 1
                        count -= 1

                    index_1 = 2 # change
                    index_2 = 1 # change
                    index_3 = 2 # change
                    index_4 = 2 # change
                    index_5 = 2 # change
                    index_6 = 2 # change
                    #print(group_number) #65

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] = \
                    dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number-1)].items(), 0, index_1))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number)].items(), 0, index_2)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+1)].items(), 0, index_3)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+2)].items(), 0, index_4)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+3)].items(), 0, index_5)))

                    newly_add_pods_dict["Pod {0}".format(new_group_number)] \
                    .update(dict(itertools.islice(language_pod_dictionary["Pod {0}".format(group_number+4)].items(), 0, index_6)))

                    #next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                    #next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                    #next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                    #next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+2)])
                    #next_start_5 = next_start_4 + len(language_pod_dictionary["Pod {0}".format(group_number+3)])
                    #for x in range(index_1):
                        #language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    #for x in range(index_2):
                        #language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                    #for x in range(index_3):
                        #language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                    
                    #for x in range(index_4):
                        #print(x+next_start_3+1)
                        #language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                    #for x in range(index_5):
                        #language_pod_dictionary["Pod {0}".format(group_number+3)].pop(name + " Student {0}".format(x+next_start_4+1))

                    #for x in range(index_6):
                        #language_pod_dictionary["Pod {0}".format(group_number+4)].pop(name + " Student {0}".format(x+next_start_5+1))

                    keys_to_remove_1 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number-1)]): 
                        if index < index_1:
                            keys_to_remove_1.append(key)
                            #print(key)
                    for key in keys_to_remove_1:
                        del language_pod_dictionary["Pod {0}".format(group_number-1)][key]

                            
                    keys_to_remove_2 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number)]): 
                        if index < index_2:
                            keys_to_remove_2.append(key)
                            #print(key)
                    for key in keys_to_remove_2:
                        del language_pod_dictionary["Pod {0}".format(group_number)][key]


                    keys_to_remove_3 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+1)]): 
                        if index < index_3:
                            keys_to_remove_3.append(key)
                            #print(key)
                    for key in keys_to_remove_3:
                        del language_pod_dictionary["Pod {0}".format(group_number+1)][key]


                    keys_to_remove_4 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+2)]): 
                        if index < index_4:
                            keys_to_remove_4.append(key)
                            #print(key)
                    for key in keys_to_remove_4:
                        del language_pod_dictionary["Pod {0}".format(group_number+2)][key]

                    
                    keys_to_remove_5 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+3)]): 
                        if index < index_5:
                            keys_to_remove_5.append(key)
                            #print(key)
                    for key in keys_to_remove_5:
                        del language_pod_dictionary["Pod {0}".format(group_number+3)][key]

                    
                    keys_to_remove_6 = []
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(group_number+4)]): 
                        if index < index_6:
                            keys_to_remove_6.append(key)
                            #print(key)
                    for key in keys_to_remove_6:
                        del language_pod_dictionary["Pod {0}".format(group_number+4)][key]
                

        return language_pod_dictionary

    


    #print("\nEnglish Groups: \n", english_pod_dictionary)
    #print("\nEnglish Extra Students: \n", extra_english_students_dict)

    #print("\nSpanish Groups: \n", spanish_pod_dictionary)
    #print("\nSpanish Extra Students: \n", extra_spanish_students_dict)

    #print("\nOther Groups: \n", other_pod_dictionary)
    #print("\nOther Extra Students: \n", extra_other_students_dict)

    #print("\nNew Groups: \n", newly_add_pods_dict)

    # COMPUTING GENDER RATIO
    #---------------------------------------------------------------------
    total_pods = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary)

    def calculate_gender_ratio(self,language_pod_dictionary, pod_number, name):
        # get the gender values from each student 
        # if "F" add 1 to female counter 
        # else if "M" add 1 to male counter
        female_count = 0
        male_count = 0
        other_count = 0
        i = 0

        if name == "English":
            student_dict_id = pod_number

        elif name == "All":
            student_dict_id = pod_number

        elif name == "Spanish":
            student_dict_id = pod_number - len(self.english_pod_dictionary)
            

        elif name == "Other":
            student_dict_id = pod_number - (len(self.spanish_pod_dictionary) + len(self.english_pod_dictionary))
        
        elif name == "New":
            student_dict_id = pod_number - (len(self.spanish_pod_dictionary) + len(self.english_pod_dictionary) + len(self.other_pod_dictionary))
        

        total_students = len(language_pod_dictionary["Pod {0}".format(pod_number)])
        original_student_counter = student_dict_id*len(language_pod_dictionary["Pod {0}".format(pod_number)])
        student_counter = student_dict_id*len(language_pod_dictionary["Pod {0}".format(pod_number)])

        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
        #while i != len(language_pod_dictionary["Pod {0}".format(pod_number)]): #name + " Student {0}".format(student_counter)
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                female_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break
            elif language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                male_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break
        
            else:
                other_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break

            i += 1


        ratio_string = f"{male_count}:{female_count}:{other_count} (M:F:O)"
        gender_ratio_double = round(male_count/total_students,2)

        print(f"Student Gender Ratio: {ratio_string}")
        #print(f"M: {male_count}")
        #print(f"F: {female_count}")
        #print(f"Other: {other_count}")

        return gender_ratio_double




    #print("English Example (Pod 1): " + str(calculate_gender_ratio(english_pod_dictionary, 1, "English")) + " (M:F)")
    #print("\n\n")
    #print("Spanish Example (Pod 64): " + str(calculate_gender_ratio(spanish_pod_dictionary, 64, "Spanish")) + " (M:F)")
    #print("\n\n")
    #print("Other Example (Pod 66): " + str(calculate_gender_ratio(other_pod_dictionary, 66, "Other")) + " (M:F)")

    # COMPUTING SCHOOL RATIO
    #---------------------------------------------------------------------
    def get_school_0(self,language_pod_dictionary, pod_number):
        school_0_count = 0
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 0:
                school_0_count += 1

        return school_0_count

    def get_school_1(self,language_pod_dictionary, pod_number):
        school_1_count = 0
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 1:
                school_1_count += 1

        return school_1_count

    def get_school_3(self,language_pod_dictionary, pod_number):
        school_3_count = 0
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 3:
                school_3_count += 1

        return school_3_count

    def get_school_5(self,language_pod_dictionary, pod_number):
        school_5_count = 0
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 5:
                school_5_count += 1

        return school_5_count

    def calculate_school_ratio(self,language_pod_dictionary, pod_number, name):
        school_0_count = 0
        school_1_count = 0
        school_3_count = 0
        school_5_count = 0


        if name == "English":
            student_dict_id = pod_number

        elif name == "All":
            student_dict_id = pod_number

        elif name == "Spanish":
            student_dict_id = pod_number - len(self.english_pod_dictionary)
            

        elif name == "Other":
            student_dict_id = pod_number - (len(self.spanish_pod_dictionary) + len(self.english_pod_dictionary))
        
        elif name == "New":
            student_dict_id = pod_number - (len(self.spanish_pod_dictionary) + len(self.english_pod_dictionary) + len(self.other_pod_dictionary))

        total_students = len(language_pod_dictionary["Pod {0}".format(pod_number)])
        original_student_counter = student_dict_id*len(language_pod_dictionary["Pod {0}".format(pod_number)])
        student_counter = student_dict_id*len(language_pod_dictionary["Pod {0}".format(pod_number)])

        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): 
        #while i != len(language_pod_dictionary["Pod {0}".format(pod_number)]): #name + " Student {0}".format(student_counter)
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 0:
                school_0_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break
            elif language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 1:
                school_1_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break
            elif language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 3:
                school_3_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break
        
            elif language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == 5:
                school_5_count += 1
                student_counter -= 1
                if (original_student_counter - student_counter) >= total_students:
                    break


        ratio_string = f"{school_0_count}:{school_1_count}:{school_3_count}:{school_5_count} (0:1:3:5)"
        school_0_ratio = f"School 0: {str(round(school_0_count/total_students,2))} ({str(round((school_0_count/total_students)*100))}%) \n" 
        school_1_ratio = f"School 1: {str(round(school_1_count/total_students,2))} ({str(round((school_1_count/total_students)*100))}%) \n" 
        school_3_ratio = f"School 3: {str(round(school_3_count/total_students,2))} ({str(round((school_3_count/total_students)*100))}%) \n" 
        school_5_ratio = f"School 5: {str(round(school_5_count/total_students,2))} ({str(round((school_5_count/total_students)*100))}%)"
        school_ratio = school_0_ratio + school_1_ratio + school_3_ratio + school_5_ratio

        print(f"Student School Ratio: {ratio_string}")
        #print(school_ratio)
        #print(school_0_ratio)
        #print(school_1_ratio)
        #print(school_3_ratio)
        #print(school_5_ratio)

        #find standard deviation
        mean = (school_0_count + school_1_count + school_3_count + school_5_count)/4
        num1 = (school_0_count - mean)**2
        num2 = (school_1_count - mean)**2
        num3 = (school_3_count - mean)**2
        num4 = (school_5_count - mean)**2
        new_mean = (num1 + num2 + num3 + num4)/4

        school_standard_deviation = sqrt(new_mean)
        print(f"School Standard Deviation: {round(school_standard_deviation,2)} \n\n")


        return school_standard_deviation

    print("\nPod Gender and School Ratios:" )
    def show_ratios(self):
        for x in range(len(self.english_pod_dictionary)):
            print(f"Pod {x+1} (English): ")
            print(str(self.calculate_gender_ratio(self.english_pod_dictionary, x+1, "English")) + " (M:F)\n")
            (str(self.calculate_school_ratio(self.english_pod_dictionary, x+1, "English")) + "\n\n")

        for x in range(len(self.spanish_pod_dictionary)):
            print(f"Pod {x+len(self.english_pod_dictionary)+1} (Spanish): ")
            print(str(self.calculate_gender_ratio(self.spanish_pod_dictionary, x+len(self.english_pod_dictionary)+1, "Spanish")) + " (M:F)\n")
            (str(self.calculate_school_ratio(self.spanish_pod_dictionary, x+len(self.english_pod_dictionary)+1, "Spanish")) + "\n\n")

        for x in range(len(self.other_pod_dictionary)):
            print(f"Pod {x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+1} (Other): ")
            print(str(self.calculate_gender_ratio(self.other_pod_dictionary, x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+1, "Other")) + " (M:F)\n")
            (str(self.calculate_school_ratio(self.other_pod_dictionary, x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+1, "Other")) + "\n\n")

        for x in range(len(self.newly_add_pods_dict)):
            print(f"Pod {x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+len(self.other_pod_dictionary)+1} (New Groups): ")
            print(str(self.calculate_gender_ratio(self.newly_add_pods_dict, x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+len(self.other_pod_dictionary)+1, "New")) + " (M:F)\n")
            (str(self.calculate_school_ratio(self.newly_add_pods_dict, x+len(self.spanish_pod_dictionary)+len(self.english_pod_dictionary)+len(self.other_pod_dictionary)+1, "New")) + "\n\n")

    #show_ratios()
    # FIXING GENDER RATIO (IF NEEDED)
    #---------------------------------------------------------------------
    def shuffle_genders(self,language_pod_dictionary, pod_number, name):
        gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, name)
        male_key = ""
        female_key = ""
        

        if gender_ratio < 0.4:
            #swap a male from pod_number+1 for a female from pod_number
            for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                    female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                    female_key = key
                    break


            for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number+1)]):
                if language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Gender"] == "M":
                    male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number+1)].items(), index, index+1))
                    male_key = key
                    break
            
            if len(str(male_key)) > 0:
                language_pod_dictionary["Pod {0}".format(pod_number)].update(male_student)
                language_pod_dictionary["Pod {0}".format(pod_number)].pop(female_key)

                language_pod_dictionary["Pod {0}".format(pod_number+1)].update(female_student)
                language_pod_dictionary["Pod {0}".format(pod_number+1)].pop(male_key)
            else:
                print("no more males")
                return False




        if gender_ratio > 0.6:
            #swap a male from pod_number+1 for a female from pod_number
            for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                    male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                    male_key = key
                    break

            for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number+1)]):
                if language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Gender"] == "F":
                    female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number+1)].items(), index, index+1))
                    female_key = key
                    break


            if len(str(female_key)) > 0:
                language_pod_dictionary["Pod {0}".format(pod_number)].update(female_student)
                language_pod_dictionary["Pod {0}".format(pod_number)].pop(male_key)

                language_pod_dictionary["Pod {0}".format(pod_number+1)].update(male_student)
                language_pod_dictionary["Pod {0}".format(pod_number+1)].pop(female_key)
            else:
                print("no more females")
                return False

        



        new_gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, name)



        return new_gender_ratio

        #print("\n")
        #print(female_student)
        #print(male_student)

    def fix_gender_ratio(self,language_pod_dictionary, name):
        i=0
        if name == "English":
            pod_count = 1
        if name == "Spanish":
            pod_count = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_count = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1

        while i != len(language_pod_dictionary)-1: #if no -1 then will give keyerror (last group wont be fixed)
            gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_count, name)
            while True:
                #reshuffle until true (recursion)(brute-force methd)
                if (gender_ratio < 0.4 or gender_ratio > 0.6):
                    gender_ratio = self.shuffle_genders(language_pod_dictionary, pod_count, name)
                    if self.shuffle_genders(language_pod_dictionary, pod_count, name) == False:
                        break

                    
                else:
                    #print("Pod {0}".format(pod_count))
                    #print(gender_ratio)
                    break
            i += 1
            pod_count += 1
                

    #print("BEFORE:")
    #shuffle_genders(english_pod_dictionary,20,"English")
    #print("AFTER:")
    #calculate_gender_ratio(english_pod_dictionary,20,"English")



    # function to deal with end pods if gender ratio < 0.4 or > 0.6 take from pods with 6:6 to make them 5:7
    def final_pod_gender_redistributor(self,language_pod_dictionary, pod_number, name):
        last_gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, name)
        i = 0

        if name == "English":
            pod_counter = 1
            last_pod_number = len(self.english_pod_dictionary)
        elif name == "Spanish":
            pod_counter = len(self.english_pod_dictionary) + 1
            last_pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary)
        elif name == "Other":
            pod_counter = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1
            last_pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary)

        if last_gender_ratio < 0.4: # less males
            while i != len(language_pod_dictionary):
                gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_counter, name)
                if gender_ratio == 0.5: #if a pod has 6:6 ratio (realistically only works for english pods)
                    # take a male out and put into last pod
                    # take a female out of last pod and put into 6:6 ratio pod
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_counter)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_counter)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_counter)].items(), index, index+1))
                            male_key = key
                            break

                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(last_pod_number)]):
                        if language_pod_dictionary["Pod {0}".format(last_pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(last_pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    if len(str(female_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_counter)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(pod_counter)].pop(male_key)

                        language_pod_dictionary["Pod {0}".format(last_pod_number)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].pop(female_key)
                        break
                    else:
                        print("no more females")
                        return False

                elif gender_ratio >= 0.4 and gender_ratio <= 0.6: # between 0.4 and 0.6, inclusive (for edge cases (spanish, other, etc))
                    # take a male out (pod_counter) and put into last pod (last_pod_number)
                    # take a female out of last pod (last_pod_number) and put into already balanced ratio pod (pod_counter) 
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_counter)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_counter)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_counter)].items(), index, index+1))
                            male_key = key
                            break

                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(last_pod_number)]):
                        if language_pod_dictionary["Pod {0}".format(last_pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(last_pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    if len(str(female_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_counter)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(pod_counter)].pop(male_key)

                        language_pod_dictionary["Pod {0}".format(last_pod_number)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].pop(female_key)
                        break
                    else:
                        print("no more females")
                        return False

                    


                pod_counter+=1
                i+=1



        if last_gender_ratio > 0.6: # more males
            while i != len(language_pod_dictionary)-1:
                gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_counter, name)
                if gender_ratio == 0.5: #if a pod has 6:6 ratio
                    # take a female out and put into last pod
                    # take a male out of last pod and put into 6:6 ratio pod
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_counter)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_counter)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_counter)].items(), index, index+1))
                            female_key = key
                            break

                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(last_pod_number)]):
                        if language_pod_dictionary["Pod {0}".format(last_pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(last_pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    if len(str(male_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].pop(male_key)

                        language_pod_dictionary["Pod {0}".format(pod_counter)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(pod_counter)].pop(female_key)
                        break
                    else:
                        print("no more males")
                        return False

                elif gender_ratio >= 0.4 and gender_ratio <= 0.6: # in between 0.4 and 0.6, inclusive
                    # take a female out and put into last pod
                    # take a male out of last pod and put into already balanced ratio pod
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_counter)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_counter)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_counter)].items(), index, index+1))
                            female_key = key
                            break

                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(last_pod_number)]):
                        if language_pod_dictionary["Pod {0}".format(last_pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(last_pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    if len(str(male_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(last_pod_number)].pop(male_key)

                        language_pod_dictionary["Pod {0}".format(pod_counter)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(pod_counter)].pop(female_key)
                        break
                    else:
                        print("no more males")
                        return False




                    

                pod_counter+=1
                i+=1
            



    def final_pod_gender_redistribution(self, language_pod_dictionary, name):
        if name == "English":
            last_pod_number = len(self.english_pod_dictionary)
        elif name == "Spanish":
            last_pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary)
        elif name == "Other":
            last_pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary)


        gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, last_pod_number, name)
        if (gender_ratio < 0.4 or gender_ratio > 0.6):
           self.final_pod_gender_redistributor(language_pod_dictionary, last_pod_number, name)
           if self.final_pod_gender_redistributor(language_pod_dictionary, last_pod_number, name) == False:
                print("error")
            


    # fix new group ratios
    #--------------------------------------------------------------
    # determine what language group
    def get_newgroup_language(self,language_pod_dictionary, pod_number):
        for index, value in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]):
            language = str(language_pod_dictionary["Pod {0}".format(pod_number)][value]["Description_HL"])
            break

        if language == "English":
            language_group = "English"

        elif language == "Spanish":
            language_group = "Spanish"

        else:
            language_group = "Other"


        return language_group

    def redistribute_new_group(self,language_pod_dictionary, pod_number, name):
        newgroup_gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, name)
        i = 0

        if self.get_newgroup_language(language_pod_dictionary, pod_number) == "English":
            other_pod_number = 1
            name = "English"
            other_language_pod_dictionary = self.english_pod_dictionary
        if self.get_newgroup_language(language_pod_dictionary, pod_number) == "Spanish":
            other_pod_number = len(self.english_pod_dictionary) + 1
            name = "Spanish"
            other_language_pod_dictionary = self.spanish_pod_dictionary
        if self.get_newgroup_language(language_pod_dictionary, pod_number) == "Other":
            other_pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1
            name = "Other"
            other_language_pod_dictionary = self.other_pod_dictionary

        if newgroup_gender_ratio < 0.4:
            while i != len(other_language_pod_dictionary):
                gender_ratio = self.calculate_gender_ratio(other_language_pod_dictionary, other_pod_number, name)
                if gender_ratio == 0.5:
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    for index, key in enumerate(other_language_pod_dictionary["Pod {0}".format(other_pod_number)]):
                        if other_language_pod_dictionary["Pod {0}".format(other_pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(other_language_pod_dictionary["Pod {0}".format(other_pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    if len(str(male_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_number)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(pod_number)].pop(female_key)

                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].update(female_student)
                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].pop(male_key)
                        break
                    else:
                        print("no more males")
                        return False

                elif gender_ratio >= 0.4 and gender_ratio <= 0.6: # between 0.4 and 0.6, inclusive (for edge cases (spanish, other, etc))
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    for index, key in enumerate(other_language_pod_dictionary["Pod {0}".format(other_pod_number)]):
                        if other_language_pod_dictionary["Pod {0}".format(other_pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(other_language_pod_dictionary["Pod {0}".format(other_pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    if len(str(male_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_number)].update(male_student)
                        language_pod_dictionary["Pod {0}".format(pod_number)].pop(female_key)

                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].update(female_student)
                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].pop(male_key)
                        break
                    else:
                        print("no more males")
                        return False
                else:
                    other_pod_number+=1
                    i+=1
                    if i == len(other_language_pod_dictionary):
                        return False

                

                
        if newgroup_gender_ratio > 0.6:
            while i != len(other_language_pod_dictionary):
                gender_ratio = self.calculate_gender_ratio(other_language_pod_dictionary, other_pod_number, name)
                if gender_ratio == 0.5:
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    for index, key in enumerate(other_language_pod_dictionary["Pod {0}".format(other_pod_number)]):
                        if other_language_pod_dictionary["Pod {0}".format(other_pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(other_language_pod_dictionary["Pod {0}".format(other_pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    if len(str(female_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_number)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(pod_number)].pop(male_key)

                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].update(male_student)
                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].pop(female_key)
                        break
                    else:
                        print("no more females")
                        return False

                elif gender_ratio >= 0.4 and gender_ratio <= 0.6: # between 0.4 and 0.6, inclusive (for edge cases (spanish, other, etc))
                    for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                        if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                            male_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                            male_key = key
                            break

                    for index, key in enumerate(other_language_pod_dictionary["Pod {0}".format(other_pod_number)]):
                        if other_language_pod_dictionary["Pod {0}".format(other_pod_number)][str(key)]["Gender"] == "F":
                            female_student = dict(itertools.islice(other_language_pod_dictionary["Pod {0}".format(other_pod_number)].items(), index, index+1))
                            female_key = key
                            break

                    if len(str(female_key)) > 0:
                        language_pod_dictionary["Pod {0}".format(pod_number)].update(female_student)
                        language_pod_dictionary["Pod {0}".format(pod_number)].pop(male_key)

                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].update(male_student)
                        other_language_pod_dictionary["Pod {0}".format(other_pod_number)].pop(female_key)
                        break
                    else:
                        print("no more females")
                        return False

                else:
                    other_pod_number+=1
                    i+=1
                    if i == len(other_language_pod_dictionary):
                        return False





        new_gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, "New")

        return new_gender_ratio
        

    # swap from same language group
    def fix_new_group_gender_ratio(self,language_pod_dictionary):
        pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + 1
        
    
        i = 0
        while i != len(language_pod_dictionary):
            gender_ratio = self.calculate_gender_ratio(language_pod_dictionary, pod_number, "New")
            while True:
                if (gender_ratio < 0.4 or gender_ratio > 0.6):
                    gender_ratio = self.redistribute_new_group(language_pod_dictionary, pod_number, "New")
                    if self.redistribute_new_group(language_pod_dictionary, pod_number, "New") == False:
                        print("skip")
                        break
                else:
                    break
                

            pod_number+=1
            i+=1
        

    


    # FIXING SCHOOL RATIO (IF NEEDED) 
    #---------------------------------------------------------------------
    def redistribute_schools(self,language_pod_dictionary, pod_number, name):
        # find which school is the most and least dominant
        # add the least dominant from (language_pod_dictionary+1) and remove most dominant from (language_pod_dictionary)
        # have to add/remove with same gender
        old_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)

        school_0 = self.get_school_0(language_pod_dictionary, pod_number)
        school_1 = self.get_school_1(language_pod_dictionary, pod_number)
        school_3 = self.get_school_3(language_pod_dictionary, pod_number)
        school_5 = self.get_school_5(language_pod_dictionary, pod_number)

        if school_0 >= school_1 and school_0 >= school_3 and school_0 >= school_5:
            dominant_school = 0
        elif school_1 >= school_0 and school_1 >= school_3 and school_1 >= school_5:
            dominant_school = 1
        elif school_3 >= school_0 and school_3 >= school_1 and school_3 >= school_5:
            dominant_school = 3
        elif school_5 >= school_0 and school_5 >= school_1 and school_5 >= school_3:
            dominant_school = 5

        if school_0 <= school_1 and school_0 <= school_3 and school_0 <= school_5:
            least_dominant_school = 0
        elif school_1 <= school_0 and school_1 <= school_3 and school_1 <= school_5:
            least_dominant_school = 1
        elif school_3 <= school_0 and school_3 <= school_1 and school_3 <= school_5:
            least_dominant_school = 3
        elif school_5 <= school_0 and school_5 <= school_1 and school_5 <= school_3:
            least_dominant_school = 5

        dominant_student = ""
        dominant_student_key = ""
        least_dominant_student = ""
        least_student_key = ""
        
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == dominant_school and language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                dominant_student_key = key 
                break

        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number+1)]):
            if language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Last Schl"] == least_dominant_school and language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Gender"] == "M":
                least_dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number+1)].items(), index, index+1))
                least_student_key = key
                break

        
        if (len(dominant_student) != 0) and (len(dominant_student_key) !=0) and (len(least_dominant_student) != 0) and (len(least_student_key) !=0):
            language_pod_dictionary["Pod {0}".format(pod_number)].update(least_dominant_student)
            language_pod_dictionary["Pod {0}".format(pod_number)].pop(dominant_student_key)

            language_pod_dictionary["Pod {0}".format(pod_number+1)].update(dominant_student)
            language_pod_dictionary["Pod {0}".format(pod_number+1)].pop(least_student_key)
        else:
            print("cant be further optimized")
            return False
        
        

        #print(least_dominant_school)
        #print(dominant_school)

        new_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        return new_standard_deviation

    def redistribute_schools_female(self,language_pod_dictionary, pod_number, name):
        # find which school is the most and least dominant
        # add the least dominant from (language_pod_dictionary+1) and remove most dominant from (language_pod_dictionary)
        # have to add/remove with same gender
        old_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)

        school_0 = self.get_school_0(language_pod_dictionary, pod_number)
        school_1 = self.get_school_1(language_pod_dictionary, pod_number)
        school_3 = self.get_school_3(language_pod_dictionary, pod_number)
        school_5 = self.get_school_5(language_pod_dictionary, pod_number)

        if school_0 >= school_1 and school_0 >= school_3 and school_0 >= school_5:
            dominant_school = 0
        elif school_1 >= school_0 and school_1 >= school_3 and school_1 >= school_5:
            dominant_school = 1
        elif school_3 >= school_0 and school_3 >= school_1 and school_3 >= school_5:
            dominant_school = 3
        elif school_5 >= school_0 and school_5 >= school_1 and school_5 >= school_3:
            dominant_school = 5

        if school_0 <= school_1 and school_0 <= school_3 and school_0 <= school_5:
            least_dominant_school = 0
        elif school_1 <= school_0 and school_1 <= school_3 and school_1 <= school_5:
            least_dominant_school = 1
        elif school_3 <= school_0 and school_3 <= school_1 and school_3 <= school_5:
            least_dominant_school = 3
        elif school_5 <= school_0 and school_5 <= school_1 and school_5 <= school_3:
            least_dominant_school = 5

        dominant_student = ""
        dominant_student_key = ""
        least_dominant_student = ""
        least_student_key = ""
        
        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
            if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == dominant_school and language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                dominant_student_key = key 
                break

        for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number+1)]):
            if language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Last Schl"] == least_dominant_school and language_pod_dictionary["Pod {0}".format(pod_number+1)][str(key)]["Gender"] == "F":
                least_dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number+1)].items(), index, index+1))
                least_student_key = key
                break

        
        if (len(dominant_student) != 0) and (len(dominant_student_key) !=0) and (len(least_dominant_student) != 0) and (len(least_student_key) !=0):
            language_pod_dictionary["Pod {0}".format(pod_number)].update(least_dominant_student)
            language_pod_dictionary["Pod {0}".format(pod_number)].pop(dominant_student_key)

            language_pod_dictionary["Pod {0}".format(pod_number+1)].update(dominant_student)
            language_pod_dictionary["Pod {0}".format(pod_number+1)].pop(least_student_key)
        else:
            print("cant be further optimized")
            return False
        
        

        #print(least_dominant_school)
        #print(dominant_school)

        new_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        return new_standard_deviation


    def fix_school_ratio_by_males(self,language_pod_dictionary, name):
        if name == "English":
            pod_number = 1
        if name == "Spanish":
            pod_number = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1

        standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        i = 0
        while i != len(language_pod_dictionary)-1:
            standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
            while True:
                if (standard_deviation > 0.5):
                    standard_deviation = self.redistribute_schools(language_pod_dictionary, pod_number, name)
                    if self.redistribute_schools(language_pod_dictionary, pod_number, name) == False:
                        print("skipped...")
                        break
                else:
                    break
                

            pod_number+=1
            i+=1

    def fix_school_ratio_by_females(self,language_pod_dictionary, name):
        if name == "English":
            pod_number = 1
        if name == "Spanish":
            pod_number = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1

        standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        i = 0
        while i != len(language_pod_dictionary)-1:
            standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
            while True:
                if (standard_deviation > 1.5):
                    standard_deviation = self.redistribute_schools_female(language_pod_dictionary, pod_number, name)
                    if self.redistribute_schools_female(language_pod_dictionary, pod_number, name) == False:
                        print("skipped...")
                        break
                else:
                    break
                

            pod_number+=1
            i+=1


    
    def final_redistribute_male(self,language_pod_dictionary, pod_number, name):
        # search for a pod that has a standard deviation of 0
        # determine which school is most/least dominant in the pod that has the highest standard deviation
        # take the least dominant school from the pod with 0 std and add to the one with higher std
        self.calculate_school_ratio(language_pod_dictionary, pod_number, name)

        school_0 = self.get_school_0(language_pod_dictionary, pod_number)
        school_1 = self.get_school_1(language_pod_dictionary, pod_number)
        school_3 = self.get_school_3(language_pod_dictionary, pod_number)
        school_5 = self.get_school_5(language_pod_dictionary, pod_number)

        dominant_student = ""
        dominant_student_key = ""
        least_dominant_student = ""
        least_student_key = ""

        if school_0 >= school_1 and school_0 >= school_3 and school_0 >= school_5:
            dominant_school = 0
        elif school_1 >= school_0 and school_1 >= school_3 and school_1 >= school_5:
            dominant_school = 1
        elif school_3 >= school_0 and school_3 >= school_1 and school_3 >= school_5:
            dominant_school = 3
        elif school_5 >= school_0 and school_5 >= school_1 and school_5 >= school_3:
            dominant_school = 5

        if school_0 <= school_1 and school_0 <= school_3 and school_0 <= school_5:
            least_dominant_school = 0
        elif school_1 <= school_0 and school_1 <= school_3 and school_1 <= school_5:
            least_dominant_school = 1
        elif school_3 <= school_0 and school_3 <= school_1 and school_3 <= school_5:
            least_dominant_school = 3
        elif school_5 <= school_0 and school_5 <= school_1 and school_5 <= school_3:
            least_dominant_school = 5

        i = 0
        low_std_pod_number = 1
        while i != len(language_pod_dictionary):
            std = self.calculate_school_ratio(language_pod_dictionary,pod_number,name)
            if std == 0:
                for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                    if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == dominant_school and language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "M":
                        dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                        dominant_student_key = key 
                        break

                for index, key in enumerate(language_pod_dictionary["Pod {0}".format(low_std_pod_number)]):
                    if language_pod_dictionary["Pod {0}".format(low_std_pod_number)][str(key)]["Last Schl"] == least_dominant_school and language_pod_dictionary["Pod {0}".format(low_std_pod_number)][str(key)]["Gender"] == "M":
                        least_dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(low_std_pod_number)].items(), index, index+1))
                        least_student_key = key
                        break
                        
                if (len(dominant_student) != 0) and (len(dominant_student_key) !=0) and (len(least_dominant_student) != 0) and (len(least_student_key) !=0):
                    language_pod_dictionary["Pod {0}".format(pod_number)].update(least_dominant_student)
                    language_pod_dictionary["Pod {0}".format(pod_number)].pop(dominant_student_key)

                    language_pod_dictionary["Pod {0}".format(low_std_pod_number)].update(dominant_student)
                    language_pod_dictionary["Pod {0}".format(low_std_pod_number)].pop(least_student_key)
                    

                else:
                    print("cant be further optimized")
                    return False

            else:
                return False

                

            low_std_pod_number+=1
            i+=1
        

        new_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        return new_standard_deviation


    def final_redistribute_female(self,language_pod_dictionary, pod_number, name):
        # search for a pod that has a standard deviation of 0
        # determine which school is most/least dominant in the pod that has the highest standard deviation
        # take the least dominant school from the pod with 0 std and add to the one with higher std
        self.calculate_school_ratio(language_pod_dictionary, pod_number, name)

        school_0 = self.get_school_0(language_pod_dictionary, pod_number)
        school_1 = self.get_school_1(language_pod_dictionary, pod_number)
        school_3 = self.get_school_3(language_pod_dictionary, pod_number)
        school_5 = self.get_school_5(language_pod_dictionary, pod_number)

        dominant_student = ""
        dominant_student_key = ""
        least_dominant_student = ""
        least_student_key = ""

        if school_0 >= school_1 and school_0 >= school_3 and school_0 >= school_5:
            dominant_school = 0
        elif school_1 >= school_0 and school_1 >= school_3 and school_1 >= school_5:
            dominant_school = 1
        elif school_3 >= school_0 and school_3 >= school_1 and school_3 >= school_5:
            dominant_school = 3
        elif school_5 >= school_0 and school_5 >= school_1 and school_5 >= school_3:
            dominant_school = 5

        if school_0 <= school_1 and school_0 <= school_3 and school_0 <= school_5:
            least_dominant_school = 0
        elif school_1 <= school_0 and school_1 <= school_3 and school_1 <= school_5:
            least_dominant_school = 1
        elif school_3 <= school_0 and school_3 <= school_1 and school_3 <= school_5:
            least_dominant_school = 3
        elif school_5 <= school_0 and school_5 <= school_1 and school_5 <= school_3:
            least_dominant_school = 5

        i = 0
        low_std_pod_number = 1
        while i != len(language_pod_dictionary):
            std = self.calculate_school_ratio(language_pod_dictionary,pod_number,name)
            if std == 0:
                for index, key in enumerate(language_pod_dictionary["Pod {0}".format(pod_number)]): # call by index
                    if language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Last Schl"] == dominant_school and language_pod_dictionary["Pod {0}".format(pod_number)][str(key)]["Gender"] == "F":
                        dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(pod_number)].items(), index, index+1))
                        dominant_student_key = key 
                        break

                for index, key in enumerate(language_pod_dictionary["Pod {0}".format(low_std_pod_number)]):
                    if language_pod_dictionary["Pod {0}".format(low_std_pod_number)][str(key)]["Last Schl"] == least_dominant_school and language_pod_dictionary["Pod {0}".format(low_std_pod_number)][str(key)]["Gender"] == "F":
                        least_dominant_student = dict(itertools.islice(language_pod_dictionary["Pod {0}".format(low_std_pod_number)].items(), index, index+1))
                        least_student_key = key
                        break
                        
                if (len(dominant_student) != 0) and (len(dominant_student_key) !=0) and (len(least_dominant_student) != 0) and (len(least_student_key) !=0):
                    language_pod_dictionary["Pod {0}".format(pod_number)].update(least_dominant_student)
                    language_pod_dictionary["Pod {0}".format(pod_number)].pop(dominant_student_key)

                    language_pod_dictionary["Pod {0}".format(low_std_pod_number)].update(dominant_student)
                    language_pod_dictionary["Pod {0}".format(low_std_pod_number)].pop(least_student_key)
                    

                else:
                    print("cant be further optimized")
                    return False

            else:
                return False

                

            low_std_pod_number+=1
            i+=1
        

        new_standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        return new_standard_deviation

    def finalize_school_redistributions_female(self,language_pod_dictionary, name):
        if name == "English":
            pod_number = 1
        if name == "Spanish":
            pod_number = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1

        standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        i = 0
        while i != len(language_pod_dictionary):
            standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
            while True:
                if (standard_deviation > 1.3):
                    standard_deviation = self.final_redistribute_female(language_pod_dictionary, pod_number, name)
                    if self.final_redistribute_female(language_pod_dictionary, pod_number, name) == False:
                        print("skipped...")
                        break
                else:
                    break
                

            pod_number+=1
            i+=1

    def finalize_school_redistributions_male(self,language_pod_dictionary, name):
        if name == "English":
            pod_number = 1
        if name == "Spanish":
            pod_number = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1

        standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        i = 0
        while i != len(language_pod_dictionary):
            standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
            while True:
                if (standard_deviation > 1.3):
                    standard_deviation = self.final_redistribute_male(language_pod_dictionary, pod_number, name)
                    if self.final_redistribute_male(language_pod_dictionary, pod_number, name) == False:
                        print("skipped...")
                        break
                else:
                    break
                

            pod_number+=1
            i+=1

    

    def count_std_above_threshold(self,language_pod_dictionary, name):
        count = 0
        if name == "English":
            pod_number = 1
        if name == "Spanish":
            pod_number = len(self.english_pod_dictionary) + 1
        if name == "Other":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + 1
        if name == "New":
            pod_number = len(self.english_pod_dictionary) + len(self.spanish_pod_dictionary) + len(self.other_pod_dictionary) + 1

        standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
        i = 0
        while i != len(language_pod_dictionary):
            standard_deviation = self.calculate_school_ratio(language_pod_dictionary, pod_number, name)
            if (standard_deviation > 2):
                count+=1
                

            pod_number+=1
            i+=1

        return count

    def runall(self):
        pass
        #if __name__ == '__main__':
            #Thread(target = self.func1).start()
            #Thread(target = self.func2).start()


    


run = CreatePods()

group_number = 1
run.add_extra_students(run.english_pod_dictionary, group_number, run.extra_english_students_dict, run.newly_add_pods_dict, "English")

group_number += len(run.english_pod_dictionary)
run.add_extra_students(run.spanish_pod_dictionary, group_number, run.extra_spanish_students_dict, run.newly_add_pods_dict, "Spanish")

group_number += len(run.spanish_pod_dictionary)
run.add_extra_students(run.other_pod_dictionary, group_number, run.extra_other_students_dict, run.newly_add_pods_dict, "Other")

run.fix_gender_ratio(run.english_pod_dictionary, "English")
run.fix_gender_ratio(run.spanish_pod_dictionary, "Spanish")
run.fix_gender_ratio(run.other_pod_dictionary, "Other")
#show_ratios()
run.final_pod_gender_redistribution(run.english_pod_dictionary, "English")
run.final_pod_gender_redistribution(run.spanish_pod_dictionary, "Spanish")
run.final_pod_gender_redistribution(run.other_pod_dictionary, "Other")
#print("\nAfter gender redistribution:")
#run.show_ratios()

run.fix_new_group_gender_ratio(run.newly_add_pods_dict)  
#run.show_ratios()

#redistribute_schools(english_pod_dictionary, 1, "English")
run.fix_school_ratio_by_males(run.english_pod_dictionary, "English")
run.fix_school_ratio_by_males(run.spanish_pod_dictionary, "Spanish")
run.fix_school_ratio_by_males(run.other_pod_dictionary, "Other")
run.fix_school_ratio_by_females(run.english_pod_dictionary, "English")
run.fix_school_ratio_by_females(run.spanish_pod_dictionary, "Spanish")
run.fix_school_ratio_by_females(run.other_pod_dictionary, "Other")
#run.show_ratios()


run.finalize_school_redistributions_female(run.english_pod_dictionary, "English")
run.finalize_school_redistributions_female(run.spanish_pod_dictionary, "Spanish")
run.finalize_school_redistributions_female(run.other_pod_dictionary, "Other")
run.finalize_school_redistributions_male(run.english_pod_dictionary, "English")
run.finalize_school_redistributions_male(run.spanish_pod_dictionary, "Spanish")
run.finalize_school_redistributions_male(run.other_pod_dictionary, "Other")


english_over_threshold = run.count_std_above_threshold(run.english_pod_dictionary, "English")
spanish_over_threshold = run.count_std_above_threshold(run.spanish_pod_dictionary, "Spanish")
other_over_threshold = run.count_std_above_threshold(run.other_pod_dictionary, "Other")
if len(run.newly_add_pods_dict) != 0:
    new_over_threshold = run.count_std_above_threshold(run.newly_add_pods_dict, "New")

run.show_ratios()
print("# of Pods above 2 STDs: ")
print(f"English: {english_over_threshold}")
print(f"Spanish: {spanish_over_threshold}")
print(f"Other: {other_over_threshold}")
if len(run.newly_add_pods_dict) != 0:
    print(f"New: {new_over_threshold}")



all_pod_groups_dictionary = dict()

all_pod_groups_dictionary.update(run.english_pod_dictionary)
all_pod_groups_dictionary.update(run.spanish_pod_dictionary)
all_pod_groups_dictionary.update(run.other_pod_dictionary)
all_pod_groups_dictionary.update(run.newly_add_pods_dict)

number_of_total_pod_groups = len(all_pod_groups_dictionary)

i = 0
pod_counter = 1
total_students = 0
while i != len(all_pod_groups_dictionary):
    total_students += len(all_pod_groups_dictionary["Pod {0}".format(pod_counter)])
    i += 1
    pod_counter += 1

#print(total_students)

#print("\n\n",all_pod_groups_dictionary)