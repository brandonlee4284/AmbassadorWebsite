from django.contrib import admin
from django.urls import path
from django.shortcuts import render
import tensorflow
from .models import Student
from django import forms
from django.contrib import messages
from django.http import HttpResponseRedirect, response
from django.urls import reverse
import csv
from django.http import HttpResponse
import pandas as pd
import random
import tensorflow as tf
import numpy as np
from .models import Pod, Home, Resources, Schedule
import itertools

class CsvImportForm(forms.Form):
    csv_upload = forms.FileField()

class StudentAdmin(admin.ModelAdmin):
    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('upload-csv/', self.upload_csv),]
        return new_urls + urls


    

    def upload_csv(self, request):

        if request.method == "POST":
            csv_file = request.FILES["csv_upload"]
            
            if not csv_file.name.endswith('.csv'):
                messages.warning(request, 'The wrong file type was uploaded. Upload a CSV formated file')
                return HttpResponseRedirect(request.path_info)
            
            df = pd.read_csv(csv_file)
            df.to_csv('student_data.csv', index=False)
            


            url = '/admin/database/student'
            return HttpResponseRedirect(url)

        form = CsvImportForm()
        data = {"form": form}
        
        return render(request, "admin/csv_upload.html", data)

    
class PodAdmin(admin.ModelAdmin):
    list_display = ('pod_group_number', 'pod_leader', 'pod_room_number', 'pod_group_members')

    def get_urls(self):
        urls = super().get_urls()
        new_urls = [path('create-pod/', self.create_pod),
                    ]
        return new_urls + urls

    def create_pod(self, request):
        if request.method == "POST":

            # NORMALIZING TRAINING DATA
            #---------------------------------------------------------------------
            file_path_train = 'training_data.csv'

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
            file_path = 'student_data.csv'

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


            #print("\nModified Testing Data: ")
            #print(student_testing_data)


            # TESTING MODEL
            #---------------------------------------------------------------------
            student_testing_data = np.array(student_testing_data)


            print("\nTesting model: ")

            student_model.evaluate(student_testing_data, student_testing_labels, verbose=1)

            student_testing_labels = student_model.predict(student_testing_data)


            #print("\nData: \n{}".format(student_testing_data))
            #print('\nNot ELD: 0, ELD: 1') 


            #print("\nPod Group # Predictions (Not rounded): \n{}".format(abs(student_testing_labels)))
            #print('\nEnglish Group: 0, Spanish Group: 1, Mandarin Group: 2, Other Group: 3') 

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

            print(total_spanish_students)

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

            def add_extra_students(language_pod_dictionary, group_number, extra_students_dict, newly_add_pods_dict, name):
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
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                        if len(extra_students_dict) == 3:
                            # 2 groups: (7,8)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                        if len(extra_students_dict) == 4:
                            # 2 groups: (8,8)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                        if len(extra_students_dict) == 5:
                            # 2 groups: (8,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            
                        if len(extra_students_dict) == 6:
                            # 2 groups: (9,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                        if len(extra_students_dict) == 7:
                            # 2 groups: (10,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            for x in range(half_way_index):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                    # 2
                    #------------------------------------
                    if len(language_pod_dictionary) == 2: 
                
                        if len(extra_students_dict) == 3:
                            # 3 groups: (9,9,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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
                            
                            next_start = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start+1))

                        if len(extra_students_dict) == 4:
                            # 3 groups: (10,9,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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
                            
                            next_start = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start+1))

                        if len(extra_students_dict) == 5:
                            # 3 groups: (10,10,9)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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
                            
                            next_start = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start+1))

                        if len(extra_students_dict) == 6:
                            # 3 groups: (10,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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
                            
                            next_start = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start+1))

                        if len(extra_students_dict) == 7:
                            # 3 groups: (11,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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
                            
                            next_start = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start+1))

                    # 3
                    #------------------------------------
                
                    if len(language_pod_dictionary) == 3: 
            
                        if len(extra_students_dict) == 4:
                            # 4 groups: (10,10,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))

                        if len(extra_students_dict) == 5:
                            # 4 groups: (11,10,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))

                        if len(extra_students_dict) == 6:
                            # 4 groups: (11,11,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-3)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-3)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_2+1))

                        if len(extra_students_dict) == 7:
                            # 4 groups: (11,11,11,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))

                    # 4
                    #------------------------------------

                    if len(language_pod_dictionary) == 4: 
            
                        if len(extra_students_dict) == 5:
                            # 5 groups: (11,11,11,10,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                        if len(extra_students_dict) == 6:
                            # 5 groups: (11,11,11,11,10)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                #print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_3+1))



                        if len(extra_students_dict) == 7:
                            # 5 groups: (11,11,11,11,11)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-3)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-3)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                #print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_3+1))


                    # 5
                    #------------------------------------
                    
                    if len(language_pod_dictionary) == 5: 
                    
                        if len(extra_students_dict) == 6:
                            # 6 groups: (11,11,11,11,11,11)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                            next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+2)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                #print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                            for x in range(index_5):
                                language_pod_dictionary["Pod {0}".format(group_number+3)].pop(name + " Student {0}".format(x+next_start_4+1))

                        if len(extra_students_dict) == 7:
                            # 6 groups: (12,11,11,11,11,11)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-2)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-2)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                #print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_3+1))

                            for x in range(index_5):
                                language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_4+1))

                    # 6
                    #------------------------------------

                    if len(language_pod_dictionary) == 6: 
                    
                        if len(extra_students_dict) == 7:
                            # 7 groups: (12,12,11,11,11,11,11)
                            new_group_number = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary) + len(newly_add_pods_dict) + 1
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

                            next_start_1 = len(language_pod_dictionary["Pod {0}".format(group_number-1)])
                            next_start_2 = next_start_1 + len(language_pod_dictionary["Pod {0}".format(group_number)])
                            next_start_3 = next_start_2 + len(language_pod_dictionary["Pod {0}".format(group_number+1)])
                            next_start_4 = next_start_3 + len(language_pod_dictionary["Pod {0}".format(group_number+2)])
                            next_start_5 = next_start_4 + len(language_pod_dictionary["Pod {0}".format(group_number+3)])
                            for x in range(index_1):
                                language_pod_dictionary["Pod {0}".format(group_number-1)].pop(name + " Student {0}".format(x+1))

                            for x in range(index_2):
                                language_pod_dictionary["Pod {0}".format(group_number)].pop(name + " Student {0}".format(x+next_start_1+1))

                            for x in range(index_3):
                                language_pod_dictionary["Pod {0}".format(group_number+1)].pop(name + " Student {0}".format(x+next_start_2+1))
                            
                            for x in range(index_4):
                                #print(x+next_start_3+1)
                                language_pod_dictionary["Pod {0}".format(group_number+2)].pop(name + " Student {0}".format(x+next_start_3+1))

                            for x in range(index_5):
                                language_pod_dictionary["Pod {0}".format(group_number+3)].pop(name + " Student {0}".format(x+next_start_4+1))

                            for x in range(index_6):
                                language_pod_dictionary["Pod {0}".format(group_number+4)].pop(name + " Student {0}".format(x+next_start_5+1))
                        

                return language_pod_dictionary

            group_number = 1
            add_extra_students(english_pod_dictionary, group_number, extra_english_students_dict, newly_add_pods_dict, "English")


            group_number += len(english_pod_dictionary)
            add_extra_students(spanish_pod_dictionary, group_number, extra_spanish_students_dict, newly_add_pods_dict, "Spanish")


            group_number += len(spanish_pod_dictionary)
            add_extra_students(other_pod_dictionary, group_number, extra_other_students_dict, newly_add_pods_dict, "Other")


            print("\nEnglish Groups: \n", english_pod_dictionary)
            #print("\nEnglish Extra Students: \n", extra_english_students_dict)

            print("\nSpanish Groups: \n", spanish_pod_dictionary)
            #print("\nSpanish Extra Students: \n", extra_spanish_students_dict)

            print("\nOther Groups: \n", other_pod_dictionary)
            #print("\nOther Extra Students: \n", extra_other_students_dict)

            print("\nNew Groups: \n", newly_add_pods_dict)

            # COMPUTING GENDER RATIO
            #---------------------------------------------------------------------
            total_pods = len(english_pod_dictionary) + len(spanish_pod_dictionary) + len(other_pod_dictionary)

            def calculate_gender_ratio(language_pod_dictionary, pod_number, name):
                # get the gender values from each student 
                # if "F" add 1 to female counter 
                # else if "M" add 1 to male counter
                female_count = 0
                male_count = 0
                other_count = 0
                i = 0

                if name == "English":
                    student_dict_id = pod_number

                elif name == "Spanish":
                    student_dict_id = pod_number - len(english_pod_dictionary)
                    

                elif name == "Other":
                    student_dict_id = pod_number - (len(spanish_pod_dictionary) + len(english_pod_dictionary))
                
                elif name == "New":
                    student_dict_id = pod_number - (len(spanish_pod_dictionary) + len(english_pod_dictionary) + len(other_pod_dictionary))
                

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
            def calculate_school_ratio(language_pod_dictionary, pod_number, name):
                school_0_count = 0
                school_1_count = 0
                school_3_count = 0
                school_5_count = 0

                if name == "English":
                    student_dict_id = pod_number

                elif name == "Spanish":
                    student_dict_id = pod_number - len(english_pod_dictionary)
                    

                elif name == "Other":
                    student_dict_id = pod_number - (len(spanish_pod_dictionary) + len(english_pod_dictionary))
                
                elif name == "New":
                    student_dict_id = pod_number - (len(spanish_pod_dictionary) + len(english_pod_dictionary) + len(other_pod_dictionary))

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
                #print(school_0_ratio)
                #print(school_1_ratio)
                #print(school_3_ratio)
                #print(school_5_ratio)


                return school_ratio


            print("\nPod Gender and School Ratios:" )
            for x in range(len(english_pod_dictionary)):
                print(f"Pod {x+1} (English): ")
                print(str(calculate_gender_ratio(english_pod_dictionary, x+1, "English")) + " (M:F)\n")
                print(str(calculate_school_ratio(english_pod_dictionary, x+1, "English")) + "\n\n")

            for x in range(len(spanish_pod_dictionary)):
                print(f"Pod {x+len(english_pod_dictionary)+1} (Spanish): ")
                print(str(calculate_gender_ratio(spanish_pod_dictionary, x+len(english_pod_dictionary)+1, "Spanish")) + " (M:F)\n")
                print(str(calculate_school_ratio(spanish_pod_dictionary, x+len(english_pod_dictionary)+1, "Spanish")) + "\n\n")

            for x in range(len(other_pod_dictionary)):
                print(f"Pod {x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+1} (Other): ")
                print(str(calculate_gender_ratio(other_pod_dictionary, x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+1, "Other")) + " (M:F)\n")
                print(str(calculate_school_ratio(other_pod_dictionary, x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+1, "Other")) + "\n\n")

            for x in range(len(newly_add_pods_dict)):
                print(f"Pod {x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+len(other_pod_dictionary)+1} (New Groups): ")
                print(str(calculate_gender_ratio(newly_add_pods_dict, x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+len(other_pod_dictionary)+1, "New")) + " (M:F)\n")
                print(str(calculate_school_ratio(newly_add_pods_dict, x+len(spanish_pod_dictionary)+len(english_pod_dictionary)+len(other_pod_dictionary)+1, "New")) + "\n\n")


            # FIXING GENDER RATIO (IF NEEDED)
            #---------------------------------------------------------------------
            def fix_gender_ratio(language_pod_dictionary, pod_number, name):
                gender_ratio = calculate_gender_ratio(language_pod_dictionary, pod_number, name)

                if gender_ratio > 0.4:
                    #reshuffle?
                    return True
                    

                return language_pod_dictionary

            # FIXING SCHOOL RATIO (IF NEEDED)
            #---------------------------------------------------------------------
            def fix_school_ratio(language_pod_dictionary, pod_number, name):
                

                return language_pod_dictionary



            # ALL POD GROUPS TOGETHER
            #---------------------------------------------------------------------
            all_pod_groups_dictionary = dict()

            all_pod_groups_dictionary.update(english_pod_dictionary)
            all_pod_groups_dictionary.update(spanish_pod_dictionary)
            all_pod_groups_dictionary.update(other_pod_dictionary)
            all_pod_groups_dictionary.update(newly_add_pods_dict)

            number_of_total_pod_groups = len(all_pod_groups_dictionary)

            #print("\n\n",all_pod_groups_dictionary)
            
            
            i = 1
            while i <= number_of_total_pod_groups:
                created = Pod.objects.update_or_create(
                     pod_group_number = "Pod {0}".format(i),
                     pod_leader = "Pod Leader",
                     pod_room_number = random.randint(100, 999),
                     pod_group_members = all_pod_groups_dictionary["Pod {0}".format(i)]

                )
                i += 1
            



            url = '/admin/database/pod/' #success (create a path)
            return HttpResponseRedirect(url)


        return render(request, "admin/create_pod.html")


     

admin.site.register(Student, StudentAdmin)
admin.site.register(Home)
admin.site.register(Pod, PodAdmin)
admin.site.register(Schedule)
admin.site.register(Resources)

