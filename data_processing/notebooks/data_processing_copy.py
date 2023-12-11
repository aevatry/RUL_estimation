import numpy as np
import os
import pathlib
from nptdms import TdmsFile


def get_tdms_generator(path:str, group: str, channels: list):

    """
    Takes a Hdf15 object and extracts the data stored in the corresponding group and fields 

    Args:

        path: path of .tdms file from which data must be extracted
        group: string. Group where the datasets of interest are stored
        channels: list. All datasets in the group that we want to return

    Returns:

        Generator object to iterate over all lines of the datasets.
        First line yielded is "channels" 
        Each iteration returns a list of all values contained in the datasets. List has the same order as parameter "channels"

    Raises:
        TypeError: if any of the input variables are not to the correct format
    """

    if not os.path.isfile(path):
        raise FileExistsError(f"Specified file does not exist at: {path}")
    
    if not pathlib.Path(path).suffix == ".tdms":
        raise TypeError (f"The file at {path} must be a .tdms file")
    
    if not isinstance(channels, list):
        raise TypeError 
    if not isinstance(group, str):
        raise TypeError
    
    with TdmsFile.open(path) as tdms_file:

        tdsm_struct = get_structure(tdms_file)
        check_keys(tdsm_struct, group, channels)
        min_data_lenght = get_channel_size(tdms_file, group, channels)

        yield channels

        for i in range(0, min_data_lenght):
            row =[]
            for channel in channels:
                row+=[tdms_file[group][channel][i]]
            yield row
    

def get_structure(tdms_file):
    
    groups = []
    channels = []

    index = 0 # index to have sublists for the channels 
    for group in  tdms_file.groups():
        groups+= [group.name]
        channels+=[[]]

        for channel in tdms_file[group.name].channels():
            channels[index]+=[channel.name]
 
        index += 1

    return groups, channels

def check_keys(tdms_struct, group, channels):

    if not (group  in tdms_struct[0]):
        raise Exception(f"The provided group {group} is not a valid group")
    
    else:
        group_index = tdms_struct[0].index(group)
        for channel in channels:
            if not channel in tdms_struct[1][group_index]:
                raise Exception(f"The channel {channel} is not present in the group {group}")
            
def get_channel_size(tdms_file, group, channels):

    channel_sizes = []
    # return the smallest channel size.
    # also print if there is different channel lenght
    
    for channel in channels:
        channel_sizes+=[tdms_file[group][channel]._length]
        if not channel_sizes[0] == tdms_file[group][channel]._length:
            print(f"Channels: '{channels[0]}' and '{channel}' do not have the same size : {channel_sizes[0]} and {tdms_file[group][channel]._length} ")
    
    return min(channel_sizes)

#Formatting raw tdms files
def raw_format(raw_tdms_generator):

    if "__next__" not in raw_tdms_generator.__dir__():
        raise TypeError

    features = []
    label_count = 0 #will increment so that the the label column goes from 0 to the number of minutes of the test

    n = 0 #global count for statistical features

    # This is a temporary function and I know all features I want
    # It's not generalizable so not that useful

    peak_max = 0
    peak_min = 0

    xis = 0
    xis_squared = 0
    xis_cubed = 0
    xis_fourth = 0

    features += [['P to P', 'RMS', 'Kurtosis' , 'Labels']]

    row1 = next(raw_tdms_generator)
    time_index = row1.index('Time')
    voltage_index = row1.index('Voltage')

    for row in raw_tdms_generator:

        if peak_max<row[voltage_index]:
            peak_max = row[voltage_index]
        if peak_min>row[voltage_index]:
            peak_min = row[voltage_index]
        
        xis += row[voltage_index]
        xis_squared += (row[voltage_index])**2
        xis_cubed += (row[voltage_index])**3
        xis_fourth += (row[voltage_index])**4

        if len(row[time_index])>8 and n!=0:
            
            label_count += 1
            # RMS and Peak to Peak calculations
            p_to_p = peak_max - peak_min
            RMS = np.sqrt( (1/n) * xis_squared )

            # Kurtosis calculations
            mean = xis * (1/n)
            s_squared = (1/(n-1))*(xis_squared - 2*mean*xis + n*mean**2) 
            main_term = xis_fourth -4*mean*xis_cubed + 6*(mean**2)*xis_squared - 4*(mean**3)*xis + n*(mean**4)

            kurtosis = (1/((n-1)*(s_squared**s_squared)))*main_term

            features += [[p_to_p, RMS, kurtosis, label_count]]

        n+=1

    return features

def raw_format2(raw_tdms_generator):

    if "__next__" not in raw_tdms_generator.__dir__():
        raise TypeError

    features = []
    label_count = 0 #will increment so that the the label column goes from 0 to the number of minutes of the test

    n = 0 #global count for statistical features

    # This is a temporary function and I know all features I want
    # It's not generalizable so not that useful

    peak_max = 0
    peak_min = 0

    xis = 0
    xis_squared = 0
    xis_cubed = 0
    xis_fourth = 0

    features += [['P to P', 'RMS', 'Kurtosis' , 'Labels']]

    row1 = next(raw_tdms_generator)
    time_index = row1.index('Time')
    voltage_index = row1.index('Voltage')

    for row in raw_tdms_generator:

        if peak_max<row[voltage_index]:
            peak_max = row[voltage_index]
        if peak_min>row[voltage_index]:
            peak_min = row[voltage_index]
        
        xis += row[voltage_index]
        xis_squared += (row[voltage_index])**2
        xis_cubed += (row[voltage_index])**3
        xis_fourth += (row[voltage_index])**4

        if len(row[time_index])>8 and n!=0:
            
            label_count += 1
            # RMS and Peak to Peak calculations
            p_to_p = peak_max - peak_min
            RMS = np.sqrt( (1/n) * xis_squared )

            # Kurtosis calculations
            mean = xis * (1/n)
            s_squared = (1/(n-1))*(xis_squared - 2*mean*xis + n*mean**2) 
            main_term = xis_fourth -4*mean*xis_cubed + 6*(mean**2)*xis_squared - 4*(mean**3)*xis + n*(mean**4)

            kurtosis = (1/((n-1)*(s_squared**s_squared)))*main_term

            features += [[p_to_p, RMS, kurtosis, label_count]]

            n = 0
            xis = 0
            xis_squared = 0
            xis_cubed = 0
            xis_fourth = 0
            peak_max = 0
            peak_min = 0

        n+=1

    return features

def generator_mean(generator):

    if "__next__" not in generator.__dir__():
        raise TypeError("Object doesn't have __next__ in its directory: interpreted as not being a generator")
    
    sample_size = 0
    tot_sum = 0

    for element in generator:
        sample_size+=1
        tot_sum+=element
    return tot_sum/sample_size

def rul_labels(time_index, healthy_index):

    healthytime = time_index[healthy_index]

    for i in range(0, len(time_index)):
        time_index[i] = time_index[i] - healthytime
        time_index[i] =1 - time_index[i] / (len(time_index)-healthytime)
        if i<=healthy_index:
            time_index[i] = 1
    return time_index
