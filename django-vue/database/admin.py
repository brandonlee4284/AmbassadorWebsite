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
from .models import Pod, Home, Resources, Schedule, HomeImage

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
                messages.warning(request, 'The wrong file type was uploaded')
                return HttpResponseRedirect(request.path_info)
            
            df = pd.read_csv(csv_file)
            df.to_csv('/Users/brandonlee/Desktop/V3 Ambassador Project/django-vue/database/student_data.csv', index=False)
            


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
            new_model = tf.keras.models.load_model('/Users/brandonlee/Desktop/V3 Ambassador Project/ML Model/saved_student_model/my_model')

            # NORMALIZING TESTING DATA
            #---------------------------------------------------------------------
            file_path = '/Users/brandonlee/Desktop/V3 Ambassador Project/django-vue/database/student_data.csv'

            student_test_data = pd.read_csv(file_path)

            student_testing_data = student_test_data.copy()
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


            #print("\nModified Testing Data: ")
            #print(student_testing_data)


            # TESTING MODEL
            #---------------------------------------------------------------------
            student_testing_data = np.array(student_testing_data)


            #print("\nTesting model: ")

            new_model.evaluate(student_testing_data, student_testing_labels, verbose=1)

            student_testing_labels = new_model.predict(student_testing_data)


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

            #print(student_final_data)

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
                    english_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} Student ID: {student_dictionary[i]['Student ID']}")

                if student_dictionary[i]['POD GROUP'] == 1:
                    spanish_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} Student ID: {student_dictionary[i]['Student ID']}")

                if student_dictionary[i]['POD GROUP'] == 2:
                    mandarin_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} Student ID: {student_dictionary[i]['Student ID']}")

                if student_dictionary[i]['POD GROUP'] == 3:
                    other_group_name.append(f"{student_dictionary[i]['First Name']} {student_dictionary[i]['Last Name']} Student ID: {student_dictionary[i]['Student ID']}")
                    
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

            def group_students(student_dictionary, group_name, total_students, group_number):
                i = 0

                while total_students != i:
                    if i == 0 or i % 12 == 0:
                        j = i + 12
                        student_dictionary["Group {0}".format(group_number)] = group_name[i:j]
                        group_number += 1

                    i += 1

                return student_dictionary

            group_number = 1
            group_students(english_student_dictionary, english_group_name, total_english_students, group_number)
            group_number += total_english_pod_groups
            group_students(spanish_student_dictionary, spanish_group_name, total_spanish_students, group_number)
            group_number += total_spanish_pod_groups
            group_students(mandarin_student_dictionary, mandarin_group_name, total_mandarin_students, group_number)
            group_number += total_mandarin_pod_groups
            group_students(other_student_dictionary, other_group_name, total_other_students, group_number)


            #print("\nEnglish Groups: \n", english_student_dictionary)
            #print("\nSpanish Groups: \n", spanish_student_dictionary)
            #print("\nMandarin Groups: \n", mandarin_student_dictionary)
            #print("\nOther Groups: \n", other_student_dictionary)


            # FINALIZING POD GROUPS
            #---------------------------------------------------------------------
            all_pod_groups_dictionary = {}

            all_pod_groups_dictionary.update(english_student_dictionary)
            all_pod_groups_dictionary.update(spanish_student_dictionary)
            all_pod_groups_dictionary.update(mandarin_student_dictionary)
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
admin.site.register(HomeImage)
admin.site.register(Pod, PodAdmin)
admin.site.register(Schedule)
admin.site.register(Resources)

