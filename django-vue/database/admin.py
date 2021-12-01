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

            student_training_data['Description_HL'] = student_training_data['Description_HL'].replace(to_replace=['English', 'Spanish','OTH'], value=[1, 2, 0])

            # vectorizing if in ELD
            members_allowed_values = ['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'ELD'] #how many ELD groups are there?
            student_training_data.loc[~student_training_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'
            student_training_data['Group Memberships?'] = student_training_data['Group Memberships?'].replace(to_replace=['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5','ELD', 'OTH'], value=[1, 1, 1, 1, 1,1, 0])


            # cleaning data
            student_training_data.drop(student_training_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)

            #print("\nModified Training Data: ")
            #print(student_training_data)




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
            gender_allowed_values = ['M', 'F']
            student_testing_data.loc[~student_testing_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

            student_testing_data['Gender'] = student_testing_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

            # vectorizing Language
            languages_allowed_values = ['English', 'Spanish']
            student_testing_data.loc[~student_testing_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

            student_testing_data['Description_HL'] = student_testing_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'OTH'], value=[1, 2, 0])

            # vectorizing ELD
            members_allowed_values = ['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5','ELD'] #how many ELD groups are there?
            student_testing_data.loc[~student_testing_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'

            student_testing_data['Group Memberships?'] = student_testing_data['Group Memberships?'].replace(to_replace=['ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'ELD', 'OTH'], value=[1, 1, 1, 1, 1, 1, 0])

            # cleaning data
            student_testing_data.drop(student_testing_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)


            #print("\nModified Testing Data: ")
            #print(student_testing_data)


            # TESTING MODEL
            #---------------------------------------------------------------------
            student_testing_data = np.array(student_testing_data)


            #print("\nTesting model: ")

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


            # adding students to their corresponding groups
            english_group_name = []
            spanish_group_name = []
            other_group_name = []

            i = 0

            while len(student_dictionary) != i:
                if student_dictionary[i]['POD GROUP'] == 0:
                    english_group_name.append(
                        f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} \
                        Student ID: {student_dictionary[i]['Student ID']} \
                        Gender: {student_dictionary[i]['Gender']} \
                        Last School: {student_dictionary[i]['Last Schl']} \
                        Group Memberships: {student_dictionary[i]['Group Memberships?']}")

                if student_dictionary[i]['POD GROUP'] == 1:
                    spanish_group_name.append(
                        f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} \
                        Student ID: {student_dictionary[i]['Student ID']} \
                        Gender: {student_dictionary[i]['Gender']} \
                        Last School: {student_dictionary[i]['Last Schl']} \
                        Group Memberships: {student_dictionary[i]['Group Memberships?']}")


                if student_dictionary[i]['POD GROUP'] == 3:
                    other_group_name.append(
                        f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} \
                        Student ID: {student_dictionary[i]['Student ID']} \
                        Gender: {student_dictionary[i]['Gender']} \
                        Last School: {student_dictionary[i]['Last Schl']} \
                        Group Memberships: {student_dictionary[i]['Group Memberships?']}")
                    
                i += 1




            total_freshman_students = len(student_dictionary)
            total_english_students = len(english_group_name)
            total_spanish_students = len(spanish_group_name)
            total_other_students = len(other_group_name)


            total_english_pod_groups = int(total_english_students/12) + 1
            total_spanish_pod_groups = int(total_spanish_students/12) + 1
            total_other_pod_groups = int(total_other_students/12) + 1

            # Grouping pod groups by 12s
            english_student_dictionary = {}
            spanish_student_dictionary = {}
            other_student_dictionary = {}

            random.shuffle(english_group_name)
            random.shuffle(spanish_group_name)
            random.shuffle(other_group_name)

            extra_english_students = []
            extra_spanish_students = []
            extra_other_students = []

            def group_students(student_dictionary, group_name, total_students, group_number, extra_students):
                #if the group has less than 10 people put them in a seperate array -- different extra arrays for each language group
                i = 0

                while total_students != i:
                    if i == 0 or i % 12 == 0:
                        j = i + 12
                        
                        student_dictionary["Group {0}".format(group_number)] = group_name[i:j]

                        if len(student_dictionary["Group {0}".format(group_number)]) < 8 and len(student_dictionary) > 1:
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
            group_students(other_student_dictionary, other_group_name, total_other_students, group_number,extra_other_students)


            #print("\nEnglish Groups: \n", english_student_dictionary)
            #print("\nSpanish Groups: \n", spanish_student_dictionary)
            #print("\nOther Groups: \n", other_student_dictionary)

            #print("\nEnglish Groups EXTRA: \n", extra_english_students)
            #print("\nSpanish Groups EXTRA: \n", extra_spanish_students)
            #print("\nOther Groups EXTRA: \n", extra_other_students)

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

            #print("\nEnglish Groups EXTRA: \n", extra_english_students)
            #print("\nEnglish Groups: \n", english_student_dictionary)
            

            group_number += len(english_student_dictionary)
            total_spanish_pod_groups = len(spanish_student_dictionary)
            add_extra_students(spanish_student_dictionary, extra_spanish_students, group_number, total_spanish_pod_groups)

            #print("\nSpanish Groups EXTRA: \n", extra_spanish_students)
            #print("\nSpanish Groups: \n", spanish_student_dictionary)
            

            group_number += len(spanish_student_dictionary)
            total_other_pod_groups = len(other_student_dictionary)
            add_extra_students(other_student_dictionary, extra_other_students, group_number, total_other_pod_groups)
            
            #print("\nOther Groups EXTRA: \n", extra_other_students)
            #print("\nOther Groups: \n", other_student_dictionary)
            

            spanish_new_arr = []
            other_new_arr = []

            def finalize_groups_split(student_dictionary, group_number, new_array):
                #if any group has more than 16 students split them in half
                #to make 2 seperate groups in the same language group
                i = 0
                while i != len(student_dictionary):

                    if len(student_dictionary["Group {0}".format(group_number)]) > 15:
                        #split into two groups in the same language group
                        

                        if len(student_dictionary["Group {0}".format(group_number)]) % 2 == 0:
                            half_way_index = int((len(student_dictionary["Group {0}".format(group_number)]))/2)
                            new_array.append(student_dictionary["Group {0}".format(group_number)][0:half_way_index])
                            del student_dictionary["Group {0}".format(group_number)][0:half_way_index]
                        
                        if len(student_dictionary["Group {0}".format(group_number)]) % 2 != 0:
                            half_way_index = int((len(student_dictionary["Group {0}".format(group_number)]) + 1)/2)
                            new_array.append(student_dictionary["Group {0}".format(group_number)][0:half_way_index])
                            del student_dictionary["Group {0}".format(group_number)][0:half_way_index]


                    group_number += 1
                    i += 1

                return student_dictionary

            group_number = 1

            group_number += len(english_student_dictionary)
            finalize_groups_split(spanish_student_dictionary, group_number, spanish_new_arr)

            #print("\nSpanish Groups (After Split): \n", spanish_student_dictionary)
            #print("\nSpanish Groups (Not included yet): \n", spanish_new_arr)

            group_number += len(spanish_student_dictionary)
            finalize_groups_split(other_student_dictionary, group_number, other_new_arr)

            #print("\nOther Groups (After Split): \n", other_student_dictionary)
            #print("\nOther Groups (Not included yet): \n", other_new_arr)



            def readd_splitted_groups(student_dictionary, group_number, new_array):
                # re-add the new arrays of the splitted groups
                # add to last slot and assign a value
                if len(new_array) != 0:
                    student_dictionary["Group {0}".format(group_number)] = new_array[0]

                return student_dictionary



            group_number = len(english_student_dictionary) + 1 #goes to last index of dict

            group_number += len(spanish_student_dictionary) #goes to last index of dict
            readd_splitted_groups(spanish_student_dictionary, group_number, spanish_new_arr)
            #print("\nSpanish Groups (FINAL): \n", spanish_student_dictionary)



            group_number = len(english_student_dictionary) + len(spanish_student_dictionary)
            if len(other_student_dictionary) > 0 and len(spanish_new_arr) != 0: #if no group was added before(spanish group) this wont be executed
                last_index = group_number + len(other_student_dictionary)
                other_student_dictionary["Group {0}".format(last_index)] = other_student_dictionary["Group {0}".format(group_number)] 
                del other_student_dictionary["Group {0}".format(group_number)] 
                


            group_number = len(english_student_dictionary) + len(spanish_student_dictionary) + len(other_student_dictionary) + 1 #goes to last index of dict
            readd_splitted_groups(other_student_dictionary, group_number, other_new_arr)
            #print("\nOther Groups (FINAL): \n", other_student_dictionary)





            


            # FINALIZING POD GROUPS
            #---------------------------------------------------------------------
            all_pod_groups_dictionary = {}

            all_pod_groups_dictionary.update(english_student_dictionary)
            all_pod_groups_dictionary.update(spanish_student_dictionary)
            all_pod_groups_dictionary.update(other_student_dictionary)

            number_of_total_pod_groups = len(all_pod_groups_dictionary)

            #print(all_pod_groups_dictionary)

            #file_data = csv_file.read().decode("utf-8")
            #csv_data = file_data.split("\n")
            
            i = 1
            while i <= number_of_total_pod_groups:
                created = Pod.objects.update_or_create(
                     pod_group_number = "Pod {0}".format(i),
                     pod_leader = "Pod Leader",
                     pod_room_number = random.randint(100, 999),
                     pod_group_members = all_pod_groups_dictionary["Group {0}".format(i)]

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

