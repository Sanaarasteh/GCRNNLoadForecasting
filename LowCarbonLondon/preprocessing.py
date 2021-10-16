import os
import sys

import pandas as pd
import numpy as np

from datetime import datetime
from datetime import timedelta

from source.utils import get_household_complete_data, train_val_test_separator

# defining the paths to the London datasets as a dictionary
dataset_paths = {
    'household_info': '../datasets/London/informations_households.csv',
    'acorn_groups': '../datasets/London/acorn_details.csv',
    'weather_daily': '../datasets/London/weather_daily_darksky.csv',
    'weather_hourly': '../datasets/London/weather_hourly_darksky.csv',
    'holidays': '../datasets/London/uk_bank_holidays.csv',
    'daily_block': '../datasets/London/daily_dataset/daily_dataset/',
    'hh_block': '../datasets/London/halfhourly_dataset/halfhourly_dataset/'
}

# These are the blocks which contain sufficient users with pairwise correlations
# higher than 0.5; we denote these blocks as candidate blocks
candidate_blocks = [0, 19, 28, 68, 73, 83, 92, 93]


def make_block_dataset(target_block: int, add_weather: bool = False):
    """
    This function receives a block id, loads the entire data of the block, divides it into separate customers, and
    cleans the data. At last, the function separates the train, validation, and test segments, and saves each segment
    as a npy file
    :param target_block: the id of the target block
    :param add_weather: appends the weather data if True, does not append the weather data if False
    :return: None; saves 6x(number of customers) npy files corresponding to train_x, train_y, val_x, val_y, test_x,
    test_y
    """

    print(f'[*] Preparing the dataset for Block {target_block}')

    dataset = pd.read_csv(dataset_paths['hh_block'] + f'block_{target_block}.csv')
    # removing the rows containing null values
    dataset = dataset[dataset['energy(kWh/hh)'] != 'Null']
    # converting the energy values to float type
    dataset['energy(kWh/hh)'] = dataset['energy(kWh/hh)'].astype(float)

    weather_dataset = None

    if add_weather:
        print('[*] The weather data is being included')
        weather_dataset = pd.read_csv('../datasets/London/weather_half_hourly.csv')

    # finding unique users
    household_ids = list(np.unique(dataset['LCLid']))

    data_frames = []
    for household_id in household_ids:
        data_frames.append(dataset[dataset['LCLid'] == household_id])

    # trimming the users data to their common datetime intervals
    common_dates = data_frames[0]['tstp']
    for i in range(1, len(data_frames)):
        common_dates = pd.merge(common_dates, data_frames[i]['tstp'], how='inner', on='tstp')

    # if the weather data is also included, we further trim the data to common datetime of the
    # weather data
    if weather_dataset is not None:
        common_dates = pd.merge(common_dates, weather_dataset['tstp'], how='inner', on='tstp')

    if len(common_dates) > 0:
        for i in range(len(data_frames)):
            data_frames[i] = pd.merge(data_frames[i], common_dates, how='inner', on='tstp')

    # Finding the correlation of all users
    correlations = np.eye((len(household_ids)))
    potential_users1 = []
    for i in range(len(household_ids)):
        for j in range(i + 1, len(household_ids)):
            correlation = np.corrcoef(data_frames[i]['energy(kWh/hh)'].values,
                                      data_frames[j]['energy(kWh/hh)'].values)[0, 1]
            correlations[i, j] = correlation
            correlations[j, i] = correlation
            if correlation > 0.5:
                potential_users1.append((i, j))

    # finding the users with correlations higher than 0.5
    target_users = set()
    for pair in potential_users1:
        target_users.add(pair[0])
        target_users.add(pair[1])

    target_users = sorted(list(target_users))
    target_users = [household_ids[i] for i in target_users]

    # generating the refined datasets; samples and labels come from the
    # get_household_complete_data function
    samples = []
    labels = []
    skipped_dataset = []
    for i, household_id in enumerate(target_users):
        print(f'\t[*] Loading {i + 1}/{len(target_users)} household data')

        if not add_weather:
            x, y = get_household_complete_data(dataset, household_id, common_dates, 20, 2)
        else:
            x, y = get_household_complete_data(dataset, household_id, common_dates, 20, 2, weather_dataset)

        samples.append(x)
        labels.append(y)

    if len(samples) > 0:
        i = 0
        # Saving the refined datasets
        print('[*] Saving the datasets...')

        # creating necessary directories (in case they do not already exist)
        os.makedirs('../datasets/NewLondon', exist_ok=True)
        os.makedirs(f'../datasets/NewLondon/block{target_block}_npy_datasets', exist_ok=True)

        if add_weather:
            os.makedirs(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather', exist_ok=True)

        ds_length = samples[0].shape[0]
        indices = [i for i in range(ds_length)]
        np.random.shuffle(indices)
        # separating the datasets into train, validation, test datasets
        for sample, label in zip(samples, labels):
            train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_separator(sample, label, indices)

            while i in skipped_dataset:
                i += 1

            # saving the datasets
            if not add_weather:
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_train_x.npy', train_x)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_train_y.npy', train_y)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_val_x.npy', val_x)
                np.save(f'datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_val_y.npy', val_y)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_test_x.npy', test_x)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets/'
                        f'household_{target_users[i]}_test_y.npy', test_y)
            else:
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_train_x.npy', train_x)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_train_y.npy', train_y)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_val_x.npy', val_x)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_val_y.npy', val_y)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_test_x.npy', test_x)
                np.save(f'../datasets/NewLondon/block{target_block}_npy_datasets_weather/'
                        f'household_{target_users[i]}_test_y.npy', test_y)

            i += 1
    else:
        print(f'[*] No data was generated for Block {target_block}')

    del samples
    del labels


def generate_weather_dataset():
    """
    This function prepares the data associated to the weather variables. Since the weather data is available at
    hourly frequency and the energy consumption data is available at half-hourly frequency, this function performs
    a naive interpolation to equalize the frequencies of the data recordings
    :return: None; saves the modified and cleaned weather dataset
    """

    weather_dataset = pd.read_csv('../datasets/London/weather_hourly_darksky.csv')
    columns = list(weather_dataset.columns)

    # we remove the 'time' column and instead we insert the 'tstp' column for
    # consistency with the load consumption data
    columns.insert(-1, 'tstp')
    columns.remove('time')

    filled_weather_dataset = pd.DataFrame(columns=columns)

    for row_index in range(len(weather_dataset)):
        sys.stdout.flush()
        sys.stdout.write(f'{row_index} / {len(weather_dataset)}')

        row = weather_dataset.iloc[row_index]

        date = datetime.strptime(row['time'], '%Y-%m-%d %H:%M:%S')
        date = date + timedelta(minutes=30)
        date = date.strftime('%Y-%m-%d %H:%M:%S')
        date = date + '.0000000'

        new_row = dict(row)
        old_row = dict(row)

        new_row['tstp'] = date
        old_row['tstp'] = old_row['time']
        old_row['tstp'] = old_row['tstp'] + '.0000000'

        new_row.pop('time')
        old_row.pop('time')

        filled_weather_dataset = filled_weather_dataset.append(old_row, ignore_index=True)
        filled_weather_dataset = filled_weather_dataset.append(new_row, ignore_index=True)
        sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')

    filled_weather_dataset = pd.get_dummies(filled_weather_dataset, columns=['precipType', 'icon', 'summary'])
    filled_weather_dataset.to_csv('../datasets/London/weather_half_hourly.csv')


# for block in candidate_blocks:
#     make_block_dataset(block, add_weather=True)
