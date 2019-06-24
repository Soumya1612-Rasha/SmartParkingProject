#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas
import sys
file = "./in/sf_train_set.csv"
fileOriginal = "./in/SFpark_ParkingSensorData_HourlyOccupancy_20112013.csv"
print(' Load in data from CSV file "'+ str(file)+'"')
X = pandas.read_csv(file, sep=';', encoding="ISO-8859-1", dtype={"BLOCK_ID": str})
print(' Load in original data from CSV file "'+ str(fileOriginal)+'"')
Xor = pandas.read_csv(fileOriginal, sep=',', encoding="ISO-8859-1", dtype={"BLOCK_ID": str})



print("Original FILE")
print(Xor.head())
print("PM_DISTRICT_NAME values:")
print(Xor["PM_DISTRICT_NAME"].unique())

pm_district = Xor["PM_DISTRICT_NAME"].unique()
nbBlocks = 0
for district in pm_district:
    if district != 'nan':
        dfb = Xor.loc[Xor["PM_DISTRICT_NAME"]==district]
        block_ids = dfb["BLOCK_ID"].unique()
        print()
        print("{")
        print()
        print("'PM_DISTRICT_NAME' : '",district,"',")
        print()
        nbBlocks+=len(block_ids)
        print("'Number of blocks' : ",len(block_ids),",")
        print()
        print("'block_ids' : ",block_ids)
        print()
        print("}")
        print()
        
        X.loc[X["BLOCK_ID"].isin(block_ids),"BLOCK_ID"]=district

print("Total Number of Blocks : ",nbBlocks)
print("New block ids",X["BLOCK_ID"].unique())
sys.stdin.read(1)
X.to_csv("./in/sf_train_set_district.csv", sep=';')       

