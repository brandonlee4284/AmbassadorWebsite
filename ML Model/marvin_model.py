# framework - pytorch 
# split data into 2 groups - ELD and non-ELD
# run each group seperatley through the neural network 
# input - gender, school; label - Pod Group #
# two hidden layers (ReLU)
# dropout layers 
# output equal to the amount of (total students / 12)
    # if remainder round up to ((total students / 12) + 1)
    # add a counter to limit each output to only 12 students
# loss function (std_gender + std_school)
    # loss_fn = all output (each pods) std_gender + std_school

# optimizer - adam
# 50 epochs

# training data will consist of 100 labeled pod groups 
    # pod groups will have equally spread out genders and school distribution
    # 50 ELD 50 Non-ELD
# 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_path_train = 'Data.csv' # make new data
file_path_test = 'Evaluate.csv'

# NORMALIZING DATA
#---------------------------------------------------------------------
def transform_data(file_path):
    original_data = pd.read_csv(file_path)

    transformed_data = original_data.copy()

    # vectorizing Gender
    gender_allowed_values = ['M', 'F']
    transformed_data.loc[~transformed_data['Gender'].isin(gender_allowed_values), 'Gender'] = 'OTH'

    transformed_data['Gender'] = transformed_data['Gender'].replace(to_replace=['M', 'F', 'OTH'], value=[1, 2, 0])

    # vectorizing Language
    languages_allowed_values = ['English', 'Spanish']
    transformed_data.loc[~transformed_data['Description_HL'].isin(languages_allowed_values), 'Description_HL'] = 'OTH'

    transformed_data['Description_HL'] = transformed_data['Description_HL'].replace(to_replace=['English', 'Spanish', 'OTH'], value=[1, 2, 0])

    # vectorizing if in ELD
    members_allowed_values = ['ELD', 'ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5'] #how many ELD groups are there?
    transformed_data.loc[~transformed_data['Group Memberships?'].isin(members_allowed_values), 'Group Memberships?'] = 'OTH'
    transformed_data['Group Memberships?'] = transformed_data['Group Memberships?'].replace(to_replace=['ELD','ELD 1', 'ELD 2', 'ELD 3', 'ELD 4', 'ELD 5', 'OTH'], value=[1, 1, 1, 1, 1, 1, 0])

    # cleaning data
    transformed_data.drop(transformed_data.columns.difference(['Last Schl', 'Gender','Description_HL', 'Group Memberships?', 'POD GROUP']), axis=1, inplace=True)

    return transformed_data


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = torch.nn.Linear(4,25)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(25,10)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(10,1)
        

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x


# Training data
training_data = transform_data(file_path=file_path_train)
training_labels = training_data.pop('POD GROUP')
training_labels = torch.tensor(training_labels)# pod group #
training_labels = (training_labels.reshape([len(training_data), 1]))

training_data = np.array(training_data)
training_data = torch.tensor(training_data) # gender school language eld status
training_data = training_data.type(torch.FloatTensor)

# Testing data
testing_data = transform_data(file_path=file_path_test)
testing_data.pop('POD GROUP')

testing_data = np.array(testing_data)
testing_data = torch.tensor(testing_data)
testing_data = testing_data.type(torch.FloatTensor)

#print(training_labels)
#print(total_students)

#if total_students % 12 == 0:
    #total_pods = total_students/12
#else:
    #total_pods = math.trunc(total_students/12) + 1

##print(total_pods)

model = NeuralNet().to(device)
print(model)

learning_rate = 1e-1
epochs = 100

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

#print(f"Data: {training_data} \n")
#print(f"Labels: {training_labels}")

def training_loop(training_data, training_labels, model, optimizer, loss_fn):
    #print(training_labels)
    training_data = training_data.to(device)
    training_labels = training_labels.type(torch.FloatTensor)
    training_labels = training_labels.to(device)
    prediction = model(training_data)
    loss = loss_fn(prediction, training_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {i+1}\n-----------------------")
    #print(f"Predictions {prediction}")
    print(f"Loss: {loss}\n")

def testing_loop(testing_data, model):
    testing_data = testing_data.to(device)
    predictions = model(testing_data)
    predictions.round()
    #print(f"Testing Predictions {predictions}")
   
for i in range(epochs):
    training_loop(training_data, training_labels, model, optimizer, loss_fn) 
    
testing_loop(testing_data, model)






#-----------------------------------------------------------------------------
#gender standard deviation
#females_count = 7 
#males_count = 5
#gender_mean = (males_count + females_count)/2
#num1_gender = (males_count - gender_mean)**2
#num2_gender = (females_count - gender_mean)**2
#gender_mean = (num1_gender+ num2_gender)/2
#gender_standard_deviation = math.sqrt(gender_mean)
##print(gender_standard_deviation)
##school standard deviation
#school_0_count = 4
#school_1_count = 5
#school_3_count = 7
#school_5_count = 5
#school_mean = (school_0_count + school_1_count + school_3_count + school_5_count)/4
#num1_school = (school_0_count - school_mean)**2
#num2_school = (school_1_count - school_mean)**2
#num3_school = (school_3_count - school_mean)**2
#num4_school = (school_5_count - school_mean)**2
#school_mean = (num1_school + num2_school + num3_school + num4_school)/4
#school_standard_deviation = math.sqrt(school_mean)
#print(school_standard_deviation)

#target = 0 
#loss_fn = gender_standard_deviation + school_standard_deviation
## size equals number of pods? 
#error = torch.tensor([loss_fn,2], requires_grad=True) - torch.tensor(target)
#print(error.grad_fn)
#-----------------------------------------------------------------------------




