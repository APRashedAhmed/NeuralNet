import numpy as np
import pandas as pd

def GenerateMasterList(FileDirectory):
    return  (pd.read_csv(FileDirectory, delim_whitespace = True,
                        header = None))
def GenerateMasterListWithHeader(FileDirectory):
    return  (pd.read_csv(FileDirectory, delim_whitespace = False))

def WriteTo(DataFrame, Directory):
    DataFrame.to_csv(Directory, index = False, header = None)

def WriteToWithHeader(DataFrame, Directory):
    DataFrame.to_csv(Directory, index = False)

def GenerateXandY(DataFrame, YColumns):
    NumberOfColumns = DataFrame.shape[1]

    Y = DataFrame.iloc[:,(NumberOfColumns - YColumns):]
    
    return DataFrame.iloc[:,:(NumberOfColumns - YColumns)], Y

def FindUniqueLetters(DataFrame, Column):
    return DataFrame.iloc[:][Column].unique()

def MakeBinaryAndMergeList(DataFrame, ColumnIndex, UniqueLetters):
    index = np.arange(DataFrame.shape[0])
    columns = UniqueLetters

    BinaryDF = pd.DataFrame(index=index, columns=columns)
    BinaryDF = BinaryDF.fillna(0) # with 0s rather than NaNs
    
    for i in index:
        for j in range(len(columns)):
            if DataFrame.iloc[i,ColumnIndex] == BinaryDF.columns[j]:
                BinaryDF.iloc[i,j] = 1
             
    
    return (pd.merge(DataFrame, BinaryDF, right_index = True,
                     left_index = True))

def CheckforLetters(DataFrame):
    return

def CreateDFWithRemovedLetters(DataFrame, LetterColumns):
    MergedList = DataFrame
    for i in range(len(LetterColumns)):
        UniqueLetters = FindUniqueLetters(DataFrame, LetterColumns[i])

        MergedList = MakeBinaryAndMergeList(MergedList,
                                            LetterColumns[i],
                                            UniqueLetters)

    return MergedList.drop(LetterColumns, axis=1)

def NormalizeData(DataFrame):
    index = np.arange(DataFrame.shape[0])
    columns = DataFrame.columns

    NormalizedDataFrame = pd.DataFrame(index=index, columns=columns)

    for i in range(len(DataFrame.columns)):
        mean = DataFrame.iloc[:, i].mean()
        std = DataFrame.iloc[:, i].std()

        if std == 0:
            std = 1
        
        NormalizedDataFrame.iloc[:,i] = ((DataFrame.iloc[:, i]-mean)/
                                          std)
    return NormalizedDataFrame

def TurnDFintoMatrix(DataFrame):
    return DataFrame.as_matrix(columns = None)

# solar = GenerateMasterList("DataFiles/SolarFlare_Data2.txt")

# parameters, Y = GenerateXandY(solar, 3,)

# TotalList = CreateDFWithRemovedLetters(parameters, [0, 1, 2])

# NormalizedData = NormalizeData(TotalList)
# YNormalized = NormalizeData(Y)

# WriteToWithHeader(YNormalized, "DataFiles\Data2_y_normalized.txt")
# WriteToWithHeader(TotalList, "DataFiles\Data2_raw.txt")
# WriteToWithHeader(NormalizedData, "DataFiles\Data2_normalized.txt")

Housing = GenerateMasterList("Housing\HousingData.txt")
parameters, Y = GenerateXandY(Housing, 1)
NormalizedHousing = NormalizeData(parameters)
NormalizeY = NormalizeData(Y)

WriteToWithHeader(NormalizedHousing, "Housing\HousingData_N.txt")
WriteToWithHeader(NormalizeY, "Housing\HousingY_N.txt")

# print NormalizedData.shape
# print TotalList.shape
# print parameters.shape
# print solar.shape

# print solar.iloc[:, 9].mean()
# print parameters.iloc[:,9].mean()
# print TotalList.iloc[:,6].mean()
# print NormalizedData.iloc[:,6].tail()
# print solar.tail()
