# -*- coding: utf-8 -*-
"""
Original Author: Sajjad Fouladvand; sjjd.fouladvand@gmail.com (do not hesitate to contact me if you had any questions)

All Rights Reserved.

This code converts the Truven format into an enrollee-time matrix X(E, T),
where E is the complete set of ADHD enrollees and T is the set of time points
between Jan 2009 and Dec 2015 (by month), each cell x_ij records the medications
an enrollee e_i took at time t_j. 

How to cite:
Sajjad Fouladvand, Emily R. Hankosky, Darren W. Henderson, Heather Bush, Jin Chen, Linda P. Dwoskin, Patricia R. Freeman, Kathleen Kantak, Jeffery Talbert, Shiqiang Tao, Guo-Qiang Zhang. 
Predicting Substance Use Disorder in ADHD Patients using Long-Short Term Memory Model. In 2018 IEEE International Conference on Healthcare Informatics Workshop (ICHI-W), pp. 49-50, 2018.    
    
@inproceedings{fouladvand2018predicting,
  title={Predicting Substance Use Disorder in ADHD Patients using Long-Short Term Memory Model},
  author={Fouladvand, Sajjad and Hankosky, Emily R and Henderson, Darren W and Bush, Heather and Chen, Jin and Dwoskin, Linda P and Freeman, Patricia R and Kantak, Kathleen and Talbert, Jeffery and Tao, Shiqiang and others},
  booktitle={2018 IEEE International Conference on Healthcare Informatics Workshop (ICHI-W)},
  year={2018},
  address = {New York City, New York, USA},
  organization={IEEE}
}

Some tips for the ASK-BD team:
New data is used! Methyl, Amphetamine, Modafinil and other ADHD medications are considered
Age from 0 to 30 is considered

"""

import operator
import os
import numpy as np
import pdb
from datetime import datetime
import time


#=====================================Constants are defined here
Beginnig_records=[2009,1,1]
End_of_records=[2015,12,31]
max_code=15 # The maximum code
num_patients_records=25265963#46734236#1000000#46000000 # The number of records
yearmon_ind=1 # Index to the entry in the record which shows the date of the visit
enrolid_ind=0 # Index to the entry in the record which shows the enrolment ID
CCS_661_ind=20  # Index to the entry in the record which shows is the patient is diagnosed with SUD during that visit or not
DOB_ind=7       # Index to the entry in the record which shows the patient's date of birth
GPI_start_ind=8  # DEfine where the medication codes start in the record
GPI_end_ind=19
GPI_other_ind=[8,9,10] # Index to other ADHD medications
GPI_Moda_ind=[11,14]  # Index to the entries which shows if the patient used Modafinil in this current record (month)
GPI_Methyl_ind=[12, 13] # Methylphenidate
GPI_Amph_ind=[15, 16, 17, 18, 19] #Amphetamine
init_calc_treshold=6 
sex_ind=5   
months_covered=360 # Width of the window you want to look at. We are looking at ages between 0-30 ==> 30*12=360 months. In fact, we look at the first 360 months of enrollees' life times
age_index=4
fls_temp=np.array([0,0,0,0]) # This array shows which medication category is used ([Other, Modafinil, Methylphenidate, Amphetamine])
#===============================================================This is a simple function to calculate the time difference (based on number of months) between two given dates
def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

line_drug=[] 
filtered_records=[]
sequence_temp=[]
num_patients_total = 0
num_initiation_6to12, num_initiation_6to12_positives, num_initiation_6to12_negatives=0, 0, 0
num_initiation_13to20, num_initiation_13to20_positives, num_initiation_13to20_negatives=0, 0, 0
num_initiation_21to26, num_initiation_21to26_positives, num_initiation_21to26_negatives=0, 0, 0
num_initiation_0to5, num_initiation_0to5_positives, num_initiation_0to5_negatives, num_empty_records = 0, 0, 0, 0
num_initiation_27more, num_initiation_27more_positive, num_initiation_27more_negative, num_current_label_sud = 0, 0, 0, 0

#Defining files to output extracted sequences for SUD-positive enrolleess with initiation age in [6-12]
fn_positives_6to12=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_6to12.txt')
fn_positives_labels_6to12=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_labels_6to12.txt')

#Files to output extracted sequences for SUD-negative enrolleees with initation age in [6-12]
fn_negatives_6to12=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_6to12.txt')
fn_negatives_labels_6to12=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_labels_6to12.txt')


fn_positives_13to20=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_13to20.txt')
fn_positives_labels_13to20=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_labels_13to20.txt')

fn_negatives_13to20=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_13to20.txt')
fn_negatives_labels_13to20=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_labels_13to20.txt')

fn_all_13to20_male=os.path.join(os.path.dirname(__file__), 'patient_sequences_all_13to20_male.txt')
fn_all_13to20_female=os.path.join(os.path.dirname(__file__), 'patient_sequences_all_13to20_female.txt')
fn_all_labels_13to20_male=os.path.join(os.path.dirname(__file__), 'patient_sequences_all_labels_13to20_male.txt')
fn_all_labels_13to20_female=os.path.join(os.path.dirname(__file__), 'patient_sequences_all_labels_13to20_female.txt')

fn_positives_21to26=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_21to26.txt')
fn_positives_labels_21to26=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_labels_21to26.txt')

fn_negatives_21to26=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_21to26.txt')
fn_negatives_labels_21to26=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_labels_21to26.txt')

fn_initiation_ages=os.path.join(os.path.dirname(__file__), 'initiation_ages.txt')
fn_sud_diagnosises=os.path.join(os.path.dirname(__file__), 'sud_diiagnosis.txt')

fn_positives_nonADHD=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_empty.txt')
fn_negatives_nonADHD=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_empty.txt')

fn_positives_labels_nonADHD=os.path.join(os.path.dirname(__file__), 'patient_sequences_positives_labels_empty.txt')
fn_negatives_labels_nonADHD=os.path.join(os.path.dirname(__file__), 'patient_sequences_negatives_labels_empty.txt')



fn_after_sud_notcontinued_13to20=os.path.join(os.path.dirname(__file__), 'after_sud_notcontinued_13to20.txt')
fn_after_sud_notcontinued_6to12=os.path.join(os.path.dirname(__file__), 'after_sud_notcontinued_6to12.txt')

fn_after_sud_continued_13to20=os.path.join(os.path.dirname(__file__), 'after_sud_continued_13to20.txt')
fn_after_sud_continued_6to12=os.path.join(os.path.dirname(__file__), 'after_sud_continued_6to12.txt')

drug_index=0
filtered_records=[]
num_true=0
num_false=0
current_label=False
sud_detection_date=0
methyl_stats=np.zeros(shape=(months_covered,months_covered))
amphe_stats=np.zeros(shape=(months_covered,months_covered))
moda_stats=np.zeros(shape=(months_covered,months_covered),dtype= np.int8)
others_stats=np.zeros(shape=(months_covered,months_covered))
methyl_stats_negative=np.zeros(shape=(months_covered,1))
amphe_stats_negative=np.zeros(shape=(months_covered,1))
moda_stats_negative=np.zeros(shape=(months_covered,1),dtype= np.int8)
others_stats_negative=np.zeros(shape=(months_covered,1))

all_sud_detectation_dates=[]
initiation_termination_head=['ENROLID',',','initiation age',',','initiation months',',','SUD detection age',',','SUD detection months',',','termination months',',','Amphetamines True',',','Modafinil True',',','Methylphenidate True',',','Lable''\n']
termination_months=0
fn=os.path.join(os.path.dirname(__file__), 'output_records_clean_3_notNULL.csv')
start_time = time.time()
num_adhd_diag_after_intit=0
header_sequences=[]
for i in range(months_covered):
    header_sequences.append("Month_"+str(i)+", ")
header_sequences.append("Patient ID, ")
header_sequences.append("SUD detection month from age 0, ")
header_sequences.append("Age, ")
header_sequences.append("Initiation Medication-Code, ")
header_sequences.append("Initiation Age, ")
header_sequences.append("Initiation Age_month base, ")
header_sequences.append("Initiation Age_yearmon, ")
header_sequences.append("Age in the first record, ")
header_sequences.append("SUD detection yearmon,")
header_sequences.append("Sex")
header_sequences.append("\n")
not_cleans_id=[]
num_not_cleans=0
num_real_birth_cal=0
num_calc_diff_there=0
num_long_outliers=0
   
with open(fn) as drug_file, open(fn_positives_13to20, 'w') as f_positives_13to20, open(fn_all_13to20_female, 'w') as f_all_13to20_female, open(fn_all_13to20_male, 'w') as f_all_13to20_male, open(fn_negatives_13to20, 'w') as f_negatives_13to20, open(fn_positives_labels_13to20, 'w') as f_positives_labels_13to20, open(fn_all_labels_13to20_female, 'w') as f_all_labels_13to20_female , open(fn_all_labels_13to20_male, 'w') as f_all_labels_13to20_male , open(fn_negatives_labels_13to20, 'w') as f_negatives_labels_13to20, open(fn_initiation_ages, 'w') as f_initiation_ages, open(fn_sud_diagnosises, 'w') as f_sud_diagnosises, open(fn_positives_nonADHD, 'w') as f_positives_nonADHD, open(fn_negatives_nonADHD, 'w') as f_negatives_nonADHD, open(fn_positives_labels_nonADHD, 'w') as f_positives_labels_nonADHD, open(fn_negatives_labels_nonADHD, 'w') as f_negatives_labels_nonADHD:
    f_initiation_ages.write("".join(["".join(x) for x in initiation_termination_head]))
    f_positives_13to20.write("".join(["".join(x) for x in header_sequences]))
    f_negatives_13to20.write("".join(["".join(x) for x in header_sequences]))
    f_positives_nonADHD.write("".join(["".join(x) for x in header_sequences]))
    f_negatives_nonADHD.write("".join(["".join(x) for x in header_sequences]))
    next(drug_file)
    line=drug_file.readline()
    line_drug=line.split(',')
    line_drug = [int(i) for i in line_drug]
    GPI_other_ind=[GPI_other_ind[i]-GPI_start_ind for i in range(len(GPI_other_ind))]
    GPI_Moda_ind=[GPI_Moda_ind[i]-GPI_start_ind for i in range(len(GPI_Moda_ind))]
    GPI_Methyl_ind=[GPI_Methyl_ind[i]-GPI_start_ind for i in range(len(GPI_Methyl_ind))]
    GPI_Amph_ind=[GPI_Amph_ind[i]-GPI_start_ind for i in range(len(GPI_Amph_ind))]
    while drug_index < (num_patients_records-2): # Screen all records in the data base file
        #pdb.set_trace()
        if(num_patients_total % 30000 == 0 ):
            print(drug_index)
        num_multi_drug=0
        initiation_age_yearmon=0
        initiation_age_methyl=0
        initiation_age_amphe=0
        initiation_age_moda=0
        initiation_age_other=0
        patient_drug_index=0
        filtered_records.clear()
        current_enrolid = line_drug[enrolid_ind]        
        current_yearmon=line_drug[yearmon_ind]
        all_sud_detectation_dates.append(current_enrolid)
        current_patient_label=0 
        while current_enrolid == line_drug[enrolid_ind]: # In this while loop I read all records related to one patient 
            current_patient_label=current_patient_label+int(line_drug[CCS_661_ind])
            current_enrolid=line_drug[enrolid_ind]
            current_yearmon=line_drug[yearmon_ind]
            
            filtered_records.append(line_drug)
            drug_index=drug_index+1          
            line=drug_file.readline()
            line_drug=line.split(',')
            if(line==''):
                break
            line_drug = [int(i) for i in line_drug]            
            patient_drug_index=patient_drug_index+1                        
            while line_drug[yearmon_ind]==current_yearmon and line_drug[enrolid_ind]==current_enrolid:
                  current_enrolid=line_drug[enrolid_ind]
                  current_yearmon=line_drug[yearmon_ind]
                  drug_index=drug_index+1
                  line=drug_file.readline()
                  line_drug=line.split(',')
                  if(line==''):
                      break
                  line_drug = [int(i) for i in line_drug]            
                  patient_drug_index=patient_drug_index+1            
        #pdb.set_trace()
        filtered_records_sorted=sorted(filtered_records, key=operator.itemgetter(yearmon_ind)) # Sorting the records based on the date of visits
        real_birth = filtered_records_sorted[0][DOB_ind] * 100 + 1
        beg_to_firts_record=diff_month(datetime(filtered_records_sorted[0][yearmon_ind]//100,filtered_records_sorted[0][yearmon_ind]%100, 1 ),datetime(Beginnig_records[0],Beginnig_records[1], Beginnig_records[2] ))
        temp_zeros=abs(beg_to_firts_record) 
        birth_to_first_record=diff_month(datetime(filtered_records_sorted[0][yearmon_ind]//100,filtered_records_sorted[0][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 ))
        temp_zeros=abs(birth_to_first_record)
        if(birth_to_first_record>=beg_to_firts_record):
            sequence_temp.extend([-1.0]*(birth_to_first_record-beg_to_firts_record))
            sequence_temp.extend([0.0]*(beg_to_firts_record))
        else:
            sequence_temp.extend([0.0]*(birth_to_first_record)) 
        fls=filtered_records_sorted[0][GPI_start_ind:(GPI_end_ind+1)]
        max_code=pow(2,len(fls_temp))-1
        if(sum(list(operator.itemgetter(*GPI_Amph_ind)(fls)))>0): # ،Entries from 0 to 4 (5 entries in totall) of each record are all related to Amphetamine and so if at least one of these entries is 1 that means the patient use Amphetamine during this specific month (record or visit)
            fls_temp[0]=1    # fls_temp is a 3 bit array fls_temp[other medication, Methylphenidate, Amphetamine]
        else:
            fls_temp[0]=0    
        if(sum(list(operator.itemgetter(*GPI_Methyl_ind)(fls)))>0):
            fls_temp[1]=1    # fls_temp[1] is supposed indicates if current record include Methylphenidate
        else:
            fls_temp[1]=0            
        if(sum(list(operator.itemgetter(*GPI_Moda_ind)(fls)))>0):# or sum(list(operator.itemgetter(*GPI_other_ind)(fls)))>0):
            #pdb.set_trace()
            fls_temp[2]=1   # fls_temp[2] is supposed indicates if current record include other ADHD medications    
        else:
            fls_temp[2]=0
        if(sum(list(operator.itemgetter(*GPI_other_ind)(fls)))>0):
            fls_temp[3]=1    
        else:
            fls_temp[3]=0   
        fls_temp=fls_temp[::-1]
        fls=fls_temp.tolist() 
        binary_strign_temp=''.join(str(e) for e in fls)
        initiation_age=0
        initiation_age_yearmon=0
        initiation_age_methyl=0
        initiation_age_amphe=0
        initiation_age_moda=0
        initiation_age_other=0
        sud_detection_date=0
        sud_detection_yearmon=0
        sud_detection_year=0
        if (int(binary_strign_temp,2) > 0 and initiation_age==0):
            initiation_age_yearmon=filtered_records_sorted[0][yearmon_ind]
            initiation_age=filtered_records_sorted[0][age_index]
            fls_init_med= filtered_records_sorted[0][GPI_start_ind:(GPI_end_ind+1)]
            fls_Methyl=list(operator.itemgetter(*GPI_Methyl_ind)(fls_init_med))            
            binary_strign_temp_methyl=''.join(str(e) for e in fls_Methyl)
            fls_amphe=list(operator.itemgetter(*GPI_Amph_ind)(fls_init_med))            
            binary_strign_temp_amphe=''.join(str(e) for e in fls_amphe)
            fls_moda= list(operator.itemgetter(*GPI_Moda_ind)(fls_init_med))
            binary_strign_temp_moda=''.join(str(e) for e in fls_moda)
            fls_other=list(operator.itemgetter(*GPI_other_ind)(fls_init_med))
            binary_strign_temp_other=''.join(str(e) for e in fls_other)
            if(int(binary_strign_temp_methyl,2)>0):
                initiation_age_methyl=initiation_age
            elif(int(binary_strign_temp_amphe,2)>0):
                initiation_age_amphe=initiation_age
            elif(int(binary_strign_temp_moda,2)>0):
                initiation_age_moda=initiation_age
            elif(int(binary_strign_temp_other,2)>0):
                initiation_age_other=initiation_age
        sequence_temp.append(round(int(binary_strign_temp,2)/max_code , 2))
        if(int(binary_strign_temp,2) > 0):
            termination_months=diff_month(datetime(filtered_records_sorted[0][yearmon_ind]//100,filtered_records_sorted[0][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 ))
        if(filtered_records_sorted[0][CCS_661_ind]==1 and sud_detection_date==0):
            sud_detection_date=(diff_month(datetime(filtered_records_sorted[0][yearmon_ind]//100,filtered_records_sorted[0][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 )))#np.ceil((diff_month(datetime(filtered_records_sorted[0][2]//100,filtered_records_sorted[0][2]%100, 1 ),datetime(filtered_records_sorted[0][0],1, 1 ))+1)/12)
            sud_detection_year=np.ceil(sud_detection_date/12)
            sud_detection_yearmon=filtered_records_sorted[0][yearmon_ind]
            if((filtered_records_sorted[0][yearmon_ind]%100)<=months_covered):
                sud_detection_year=sud_detection_year-1
        if(filtered_records_sorted[0][CCS_661_ind]==1):
            all_sud_detectation_dates_temp=(diff_month(datetime(filtered_records_sorted[0][yearmon_ind]//100,filtered_records_sorted[0][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 )))
            all_sud_detectation_dates.append(all_sud_detectation_dates_temp)
        #pdb.set_trace()
        for j in range(len(filtered_records_sorted)-1):
            diff_birth_to_first_drug=diff_month(datetime(filtered_records_sorted[j][yearmon_ind]//100,filtered_records_sorted[j][yearmon_ind]%100, 1 ),datetime(filtered_records_sorted[j+1][yearmon_ind]//100,filtered_records_sorted[j+1][yearmon_ind]%100, 1 ))
            temp_zeros=(abs(diff_birth_to_first_drug)-1)#*16
            if(temp_zeros>0):
                sequence_temp.extend([0.0]*temp_zeros)
            fls=filtered_records_sorted[j+1][GPI_start_ind:(GPI_end_ind+1)]
            if(sum(list(operator.itemgetter(*GPI_Amph_ind)(fls)))>0):  # Amphe
                fls_temp[0]=1
            else:
                fls_temp[0]=0    
            if(sum(list(operator.itemgetter(*GPI_Methyl_ind)(fls)))>0):   # Methyl
                fls_temp[1]=1
            else:
                fls_temp[1]=0
            if(sum(list(operator.itemgetter(*GPI_Moda_ind)(fls)))>0):
                fls_temp[2]=1
            else:
                fls_temp[2]=0
            if(sum(list(operator.itemgetter(*GPI_other_ind)(fls)))>0):   # Methyl
                fls_temp[3]=1
            else:
                fls_temp[3]=0                
            fls_temp=fls_temp[::-1]
            fls=fls_temp.tolist()
            binary_strign_temp=''.join(str(e) for e in fls)     # Amphe is 01 and Methyl is 10
            sequence_temp.append(round(int(binary_strign_temp,2)/max_code , 2))
            if(int(binary_strign_temp,2) > 0):
                termination_months=diff_month(datetime(filtered_records_sorted[j+1][yearmon_ind]//100,filtered_records_sorted[j+1][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 ))      
            last_record_to_end= diff_month(datetime(End_of_records[0],End_of_records[1], End_of_records[2] ),datetime(filtered_records_sorted[j+1][yearmon_ind]//100,filtered_records_sorted[j+1][yearmon_ind]%100, 1 ))   
            if (int(binary_strign_temp,2) > 0 and initiation_age==0):
                initiation_age_yearmon=filtered_records_sorted[j+1][yearmon_ind]
                initiation_age=filtered_records_sorted[j+1][age_index]
                fls_init_med=filtered_records_sorted[j+1][GPI_start_ind:(GPI_end_ind+1)]
                fls_Methyl=list(operator.itemgetter(*GPI_Methyl_ind)(fls_init_med))            
                binary_strign_temp_methyl=''.join(str(e) for e in fls_Methyl)
                fls_amphe=list(operator.itemgetter(*GPI_Amph_ind)(fls_init_med))            
                binary_strign_temp_amphe=''.join(str(e) for e in fls_amphe)
                fls_moda= list(operator.itemgetter(*GPI_Moda_ind)(fls_init_med))
                binary_strign_temp_moda=''.join(str(e) for e in fls_moda)
                fls_other=list(operator.itemgetter(*GPI_other_ind)(fls_init_med))
                binary_strign_temp_other=''.join(str(e) for e in fls_other)                                
                if(int(binary_strign_temp_methyl,2)>0):
                    initiation_age_methyl=initiation_age
                elif(int(binary_strign_temp_amphe,2)>0):
                    initiation_age_amphe=initiation_age
                elif(int(binary_strign_temp_moda,2)>0):
                    initiation_age_moda=initiation_age
                elif(int(binary_strign_temp_other,2)>0):
                    initiation_age_other=initiation_age                                      
            if(filtered_records_sorted[j+1][CCS_661_ind]==1 and sud_detection_date==0):
                sud_detection_date=(diff_month(datetime(filtered_records_sorted[j+1][yearmon_ind]//100,filtered_records_sorted[j+1][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 )))#np.ceil((diff_month(datetime(filtered_records_sorted[j+1][2]//100,filtered_records_sorted[j+1][2]%100, 1 ),datetime(filtered_records_sorted[0][0],1, 1 ))+1)/12)
                sud_detection_year=np.ceil(sud_detection_date/12)
                sud_detection_yearmon=filtered_records_sorted[j+1][yearmon_ind]
                if((filtered_records_sorted[j+1][yearmon_ind]%100)<=init_calc_treshold):
                    sud_detection_year=sud_detection_year-1 
                    
            if(filtered_records_sorted[j+1][CCS_661_ind]==1):
                all_sud_detectation_dates_temp=(diff_month(datetime(filtered_records_sorted[j+1][yearmon_ind]//100,filtered_records_sorted[j+1][yearmon_ind]%100, 1 ),datetime(real_birth//100,real_birth%100, 1 )))
                all_sud_detectation_dates.append(all_sud_detectation_dates_temp)    
        #pdb.set_trace()       
        if(sud_detection_date!=0):
            if(initiation_age_methyl>0):
                methyl_stats[int(initiation_age_methyl), int(sud_detection_year)] += 1
            elif(initiation_age_amphe>0):
                amphe_stats[int(initiation_age_amphe), int(sud_detection_year)] += 1
            elif(initiation_age_moda>0):    
                moda_stats[int(initiation_age_moda), int(sud_detection_year)] += 1
            elif(initiation_age_other>0):    
                others_stats[int(initiation_age_other), int(sud_detection_year)] += 1
        if(sud_detection_date == 0):
            if(initiation_age_methyl> 0):
                methyl_stats_negative[int(initiation_age_methyl), 0] += 1
            elif(initiation_age_amphe> 0):
                amphe_stats_negative[int(initiation_age_amphe), 0] += 1
            elif(initiation_age_moda> 0):    
                moda_stats_negative[int(initiation_age_moda), 0] += 1
            elif(initiation_age_other> 0):    
                others_stats_negative[int(initiation_age_other), 0] += 1            
        sequence_temp_outlier=np.array(sequence_temp)
        if(np.where(sequence_temp_outlier>=0)[0][-1]-np.where(sequence_temp_outlier>=0)[0][0] >84):
            num_patients_total=num_patients_total+1
            sequence_temp=[]
            termination_months=0
            all_sud_detectation_dates=[]
            num_long_outliers=num_long_outliers+1
        if(len(sequence_temp)<months_covered):
            sequence_temp.extend([0.0]*(last_record_to_end))
            sequence_temp.extend([-1.0]*(months_covered-len(sequence_temp)))        
        if(len(sequence_temp)>months_covered+1): # 316=312 (26*12) + 1(patient code)
            num_bigger_312=num_bigger_312+1    
            del sequence_temp[months_covered:len(sequence_temp)]  # delete the history after the age of 26        
        # The last column is the number of zero
        sequence_temp.append(current_enrolid)  
        adhd_yearmon_diag=0        
        sequence_temp.append(sud_detection_date) # Appending the SUD diagnosis time step to the end of sequence
        age_temp=diff_month(datetime(Beginnig_records[0],Beginnig_records[1], Beginnig_records[2] ),datetime(real_birth//100, real_birth%100, 1 ))    
        sequence_temp.append(age_temp) #Beginnig_records[0]-filtered_records_sorted[0][0])
        if(initiation_age_methyl>0):  # Code 1 means this patient started medication with Methyl
            sequence_temp.append(1)
        elif(initiation_age_amphe>0): # Code 2 means this patient started medication with Amphe
            sequence_temp.append(2)
        elif(initiation_age_moda>0):
            sequence_temp.append(3)
        elif(initiation_age_other>0):
            sequence_temp.append(4)
        else:
            sequence_temp.append(5)
        sequence_temp.append(initiation_age)  
        if(adhd_yearmon_diag !=0):
            adhd_from_birth=diff_month(datetime(adhd_yearmon_diag//100, adhd_yearmon_diag%100, 1 ),datetime(real_birth//100,real_birth%100, 1))
        else:
            adhd_from_birth=0
        sequence_temp.append(adhd_from_birth)
        sequence_temp.append(int(initiation_age_yearmon))
        age_in_data=filtered_records_sorted[0][age_index]
        sequence_temp.append(int(age_in_data))
        sequence_temp.append(int(sud_detection_yearmon))
        current_sex=filtered_records_sorted[0][sex_ind]
        sequence_temp.append(current_sex)
        #pdb.set_trace()
        sequence_temp_ar=np.array(sequence_temp)
        zero_places=np.where(sequence_temp_ar==0)
        codes_places=np.where(sequence_temp_ar>0)
        if(zero_places[0][0]+12 > codes_places[0][0]):
            num_not_cleans = num_not_cleans + 1
            not_cleans_id.append(current_enrolid)
        if(sud_detection_date==0):
            current_label= False
        else:
            current_label=True
        # f_initiation_ages output some detail about initiation ages and initiation medications
        f_initiation_ages.write(str(current_enrolid))
        f_initiation_ages.write(',')
        f_initiation_ages.write(str(initiation_age))
        f_initiation_ages.write(',')
        f_initiation_ages.write(str(initiation_age))#-60))
        f_initiation_ages.write(',')
        f_initiation_ages.write(str(sud_detection_year)) # this will lead to -60 as sud detection date for negative samples
        f_initiation_ages.write(',')
        f_initiation_ages.write(str(sud_detection_date))#-60)) # this will lead to -60 as sud detection date for negative samples
        f_initiation_ages.write(',')
        f_initiation_ages.write(str(termination_months))
        f_initiation_ages.write(',')
        if(initiation_age_amphe>0):
            f_initiation_ages.write(str(1))
        else:
            f_initiation_ages.write(str(0))           
        f_initiation_ages.write(',')
        if(initiation_age_moda>0):
            f_initiation_ages.write(str(1))
        else:
            f_initiation_ages.write(str(0))
        f_initiation_ages.write(',')
        if(initiation_age_methyl>0):
            f_initiation_ages.write(str(1))
        else:
            f_initiation_ages.write(str(0))
        if(initiation_age_other>0):
            f_initiation_ages.write(str(1))
        else:
            f_initiation_ages.write(str(0))
        f_initiation_ages.write(',')
        if(sud_detection_date!=0):
            f_initiation_ages.write("1")
        else:
            f_initiation_ages.write("0")
        f_initiation_ages.write('\n')
        if(sud_detection_date != 0):
           num_true=num_true+1
        else:
           num_false=num_false+1          
        f_sud_diagnosises.write(','.join(map(repr, all_sud_detectation_dates[0:len(all_sud_detectation_dates)])))               
        f_sud_diagnosises.write("\n")
        if(initiation_age >=6 and initiation_age<=12): # I don't save train and test sets for 6-12 and just count them, because of the maximum number of blocks issue in python I had
           num_initiation_6to12=num_initiation_6to12+1 
           if(sud_detection_date != 0):
             num_initiation_6to12_positives=num_initiation_6to12_positives+1            
           else:
             num_initiation_6to12_negatives=num_initiation_6to12_negatives+1
        elif (initiation_age>12 and initiation_age<=20): # save train and test sets for 12-17
             num_initiation_13to20=num_initiation_13to20+1       
             if(current_sex==1):             
                f_all_13to20_male.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
                f_all_13to20_male.write("\n") 
                f_all_labels_13to20_male.write(','.join(map(repr, [1.0,0.0])))
                f_all_labels_13to20_male.write("\n")
             elif(current_sex==0):
                f_all_13to20_female.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
                f_all_13to20_female.write("\n") 
                f_all_labels_13to20_female.write(','.join(map(repr, [1.0,0.0])))
                f_all_labels_13to20_female.write("\n")                
             if(sud_detection_date != 0):
               num_initiation_13to20_positives=num_initiation_13to20_positives+1  
               f_positives_13to20.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
               f_positives_13to20.write("\n") 
               f_positives_labels_13to20.write(','.join(map(repr, [1.0,0.0])))
               f_positives_labels_13to20.write("\n")
             else:
               num_initiation_13to20_negatives=num_initiation_13to20_negatives+1  
               f_negatives_13to20.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
               f_negatives_13to20.write("\n") 
               f_negatives_labels_13to20.write(','.join(map(repr, [0.0,1.0])))
               f_negatives_labels_13to20.write("\n") 
        elif (initiation_age>20 and initiation_age<=26): # save train and test sets for 18-26
             num_initiation_21to26=num_initiation_21to26+1   
             if(sud_detection_date != 0):
               num_initiation_21to26_positives=num_initiation_21to26_positives+1  
             else:
               num_initiation_21to26_negatives=num_initiation_21to26_negatives+1  
        elif(initiation_age<=5 and initiation_age>0):
            num_initiation_0to5=num_initiation_0to5+1
            if(sud_detection_date!=0):
                num_initiation_0to5_positives=num_initiation_0to5_positives+1
            else:
                num_initiation_0to5_negatives=num_initiation_0to5_negatives+1
        elif(initiation_age==0):
             num_empty_records=num_empty_records+1
#================================================================================            
             if(sud_detection_date != 0):
               f_positives_nonADHD.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
               f_positives_nonADHD.write("\n") 
               f_positives_labels_nonADHD.write(','.join(map(repr, [1.0,0.0])))
               f_positives_labels_nonADHD.write("\n")
             else:
               f_negatives_nonADHD.write(','.join(map(repr, sequence_temp[0:len(sequence_temp)])))
               f_negatives_nonADHD.write("\n") 
               f_negatives_labels_nonADHD.write(','.join(map(repr, [0.0,1.0])))
               f_negatives_labels_nonADHD.write("\n")             
            
#================================================================================     
        elif(initiation_age>=27):
            num_initiation_27more =num_initiation_27more+1
            if(sud_detection_date!=0):
                num_initiation_27more_positive=num_initiation_27more_positive+1
            else:
                num_initiation_27more_negative=num_initiation_27more_negative+1            
        num_patients_total=num_patients_total+1
        sequence_temp=[]
        termination_months=0
        all_sud_detectation_dates=[]
drug_header='sud_at_6, sud_at_7, sud_at_8, sud_at_9, sud_at_10, sud_at_11, sud_at_12, sud_at_13, sud_at_14, sud_at_15, sud_at_16, sud_at_17, sud_at_18, sud_at_19, sud_at_20, sud_at_21, sud_at_22, sud_at_23, sud_at_24, sud_at_25, sud_at_26'
rows = np.array(['initiated_at_6', 'initiated_at_7', 'initiated_at_8','initiated_at_9', 'initiated_at_10', 'initiated_at_11','initiated_at_12', 'initiated_at_13', 'initiated_at_14','initiated_at_15', 'initiated_at_16', 'initiated_at_17','initiated_at_18', 'initiated_at_19', 'initiated_at_20','initiated_at_21', 'initiated_at_22', 'initiated_at_23','initiated_at_24', 'initiated_at_25', 'initiated_at_26'])
rows=rows.reshape(21,1)


np.savetxt('methyl_stats_all.csv',methyl_stats, delimiter=',')
np.savetxt('amphe_stats_all.csv',amphe_stats, delimiter=',')
np.savetxt('moda_stats_all.csv',moda_stats, delimiter=',')
np.savetxt('others_stats_all.csv',others_stats, delimiter=',')

np.savetxt('methyl_stats_negative.csv',methyl_stats_negative, delimiter=',', header=drug_header, fmt="%s")
np.savetxt('amphe_stats_negative.csv',amphe_stats_negative, delimiter=',', header=drug_header, fmt="%s")
np.savetxt('moda_stats_negative.csv',moda_stats_negative, delimiter=',', header=drug_header, fmt="%s")
np.savetxt('others_stats_negative.csv',others_stats_negative, delimiter=',', header=drug_header, fmt="%s")


fn_stats=os.path.join(os.path.dirname(__file__), 'stats.txt')
with open(fn_stats, 'w') as f_stats:
    f_stats.write("The number of patients is: ")
    f_stats.write(str(num_patients_total))
    f_stats.write("\n")
    f_stats.write("The number of patients with CCS_661=0 is: ")
    f_stats.write(str(num_false))
    f_stats.write("\n")
    f_stats.write("The number of patients with CCS_661=1 is: ")
    f_stats.write(str(num_true))
    f_stats.write("\n")
    f_stats.write("**************************")
    f_stats.write("\n")
    f_stats.write("The number of patients with initiation age in [6-12]: ")
    f_stats.write(str(num_initiation_6to12))    
    f_stats.write("\n")
    f_stats.write("The number of POSITIVE(CCS_661=1) patients with initiation age in [6-12]: ")
    f_stats.write(str(num_initiation_6to12_positives))    
    f_stats.write("\n")
    f_stats.write("The number of NEGATIVE(CCS_661=0) patients with initiation age in [6-12]: ")
    f_stats.write(str(num_initiation_6to12_negatives))    
    f_stats.write("\n")    
    f_stats.write("**************************")
    f_stats.write("\n")
    f_stats.write("The number of patients with initiation age in [13-20]: ")
    f_stats.write(str(num_initiation_13to20))    
    f_stats.write("\n")
    f_stats.write("The number of POSITIVE(CCS_661=1) patients with initiation age in [13-20]: ")
    f_stats.write(str(num_initiation_13to20_positives))    
    f_stats.write("\n")
    f_stats.write("The number of NEGATIVE(CCS_661=0) patients with initiation age in [13-20]: ")
    f_stats.write(str(num_initiation_13to20_negatives))    
    f_stats.write("\n")    
    f_stats.write("**************************")
    f_stats.write("\n")
    f_stats.write("The number of patients with initiation age in [21-26]: ")
    f_stats.write(str(num_initiation_21to26))    
    f_stats.write("\n")
    f_stats.write("The number of POSITIVE(CCS_661=1) patients with initiation age in [21-26]: ")
    f_stats.write(str(num_initiation_21to26_positives))    
    f_stats.write("\n")
    f_stats.write("The number of NEGATIVE(CCS_661=0) patients with initiation age in [21-26]: ")
    f_stats.write(str(num_initiation_21to26_negatives))    
    f_stats.write("\n")    
    f_stats.write("**************************")
    f_stats.write("\n")   
    f_stats.write("The number of patients with initiation age in [27 and more): ")
    f_stats.write(str(num_initiation_27more))    
    f_stats.write("\n")
    f_stats.write("The number of POSITIVE(CCS_661=1) patients with initiation age in [27 and more): ")
    f_stats.write(str(num_initiation_27more_positive))    
    f_stats.write("\n")
    f_stats.write("The number of NEGATIVE(CCS_661=0) patients with initiation age in [27 and more): ")
    f_stats.write(str(num_initiation_27more_negative))    
    f_stats.write("\n")    
    f_stats.write("**************************")
    f_stats.write("\n")   
    f_stats.write("The number of patients with initiation age in [0-5]: ")
    f_stats.write(str(num_initiation_0to5))    
    f_stats.write("\n")
    f_stats.write("The number of POSITIVE(CCS_661=1) patients with initiation age in [0-5]: ")
    f_stats.write(str(num_initiation_0to5_positives))    
    f_stats.write("\n")
    f_stats.write("The number of NEGATIVE(CCS_661=0) patients with initiation age in [0-5]: ")
    f_stats.write(str(num_initiation_0to5_negatives))    
    f_stats.write("\n")    
    f_stats.write("**************************")
    f_stats.write("\n") 
    f_stats.write("The number of empty records (patients who have never used Methyl or Amphe medications): ")
    f_stats.write(str(num_empty_records))    
    f_stats.write("\n")
    f_stats.write("**************************")
    f_stats.write("\n")
    f_stats.write("**************************")
    f_stats.write("\n") 