import os

patient_folder = os.getcwd() + '/Patients/'

def patients_info():

    info_patients = dict()
    patients = []
    
    for nfile,file in enumerate(os.listdir(patient_folder)):
        patient = str(file)
        patients.append(patient)
        info_patients[patient] = []

        with open(patient_folder+str(file), "r") as input_file:
            for line in input_file:
                line = line.strip().split(" ")

                for word in line:
                    if "Session" in word:
                        continue
                    else:
                        info_patients[patient].append(word)


    return patients, info_patients

# a = patients_info()[1]
# print(a)


def update_patient_score(cur_patient, score):

    session_count = 1

    for file in os.listdir(patient_folder):
        if file == cur_patient:

            with open(patient_folder + str(file), "r") as input_file:
                for line in input_file:
                    session_count +=1
            f = open(patient_folder + str(file), "a")
            if session_count == 1:
                f.write("Session_"+str(session_count)+": "+ str(score))
            else:
                f.write("\n" + "Session_"+str(session_count)+": "+ str(score))
            f.close()

            print(patients_info()[1])


# b = update_patient_score("Luca", 9)
# print(b)