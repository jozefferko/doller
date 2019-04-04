import dolle_code as dc
import pandas as pd
import numpy as np
import Temp as tp


base = r'/Users/jozefferko/PycharmProjects/doll'
orig_data = dc.clean_csv(
    base + r'/Ladder-machine-1.csv',
    base + r'/output.csv',
    True
)
product_path = base + r'/ProdTable.csv'
work_path = base + r'/JmgStampTrans.csv'

prod_table = pd.read_csv(product_path, sep=';', encoding="ISO-8859-1")
work_table = pd.read_csv(work_path, sep=';')
filtered = dc.filter_raw_work_table(
    work_path,
    product_path,
    1405,
    '^CF|^SW'
)

gates = [
    '0101',
    '0102',
    '0103'
]

columns_0 = [
    'Date',
    'Indgang 0101',
    'Indgang 0102',
    'Indgang 0103',
    'Indgang 0104',
    'Indgang 0105',
    'Indgang 0106',
    'Non Duplicate 0101',
    'Non Duplicate 0102',
    'Non Duplicate 0103',
    '0102 Pace',
    '0103 Pace'
]

columns_1 = [
    'StartDateTime',
    'StopDateTime',
    'SysQtyGood',
    'JobRef',
    'Name'
]

columns_2 = [
    'JobRef',
    'StartDateTime',
    'StopDateTime',
    '0101 Sum',
    '0102 Sum',
    'Expected 0102 Sum',
    'Median 0102 Init Pace',
    'Median 0102 Pace',
    'Mean 0102 Init Pace',
    'Mean 0102 Pace',
    '0103 Sum',
    'Expected 0103 Sum',
    'Median 0103 Pace',
    'Mean 0103 Pace',
    'Product'
]
columns_3 = [
    'JobRef',
    'StartDateTime',
    'StopDateTime',
    '0101 Sum',
    '0102 Sum',
    'Expected 0102 Sum',
    '0102 Ratio',
    'Median 0102 Init Pace',
    'Median 0102 Pace',
    'Mean 0102 Init Pace',
    'Mean 0102 Pace',
    '0103 Sum',
    'Expected 0103 Sum',
    '0103 Ratio',
    'Median 0103 Pace',
    'Mean 0103 Pace',
    'Product',
    'Drilling Pattern',
    'Machine',
    'No. Sections',
    'String Size',
    'String Wood',
    'Tread Size',
    'Tread Wood',
    'No. Sections'
]

# columns_3 = ['JobRef', 'StartDateTime', 'StopDateTime', '0101 Sum', '0102 Sum', 'Expected 0102 Sum', '0102 Ratio', '0103 Sum', 'Expected 0103 Sum', '0103 Ratio', 'Product']


clean_data = tp.assemble_data(orig_data, filtered, gates, columns_0)
clean_data = pd.DataFrame(clean_data, columns=columns_0)
cvs, agg = tp.gen_statistics(clean_data, filtered, gates, [columns_1, columns_2, columns_3])

cols = agg['Product'].str.split(':', expand=True)
agg.loc[:, 'Product'] = cols[5]
cols_5 = cols.iloc[:, 5].str.split('/', expand=True)
cols_5[0] = cols_5[0].apply(lambda x: x.lstrip().rstrip())
cols_5[1] = cols_5[1].apply(lambda x: x.lstrip().rstrip())
cols_5[2] = cols_5[2].apply(lambda x: x.lstrip().rstrip())
cols_5[3] = cols_5[3].apply(lambda x: x.lstrip().rstrip())


cols_5['Drilling Pattern'] = cols[0].apply(lambda x: x[:2])
cols_5['Machine'] = cols_5[0].apply(lambda x: x[:1])
# cols_5[6] = cols_5[0].apply(lambda x: x[1:])
cols_5['String Size'] = cols_5[1].apply(lambda x: x[:1])
cols_5['String Wood'] = cols_5[1].apply(lambda x: x[1:])
cols_5['Tread Size'] = cols_5[2].apply(lambda x: x[:1])
cols_5['Tread Wood'] = cols_5[2].apply(lambda x: x[1:])
cols_5['No. Sections'] = cols_5[3].apply(lambda x: x[:-1])
cols_5 = cols_5.iloc[:, 4:]
agg = agg.merge(right=cols_5, left_index=True, right_index=True)
agg = pd.DataFrame(agg, columns=columns_3)


# cvs_2.sort_values('StartDateTime', inplace=True)
# agg_2.sort_values('StartDateTime', inplace=True)
print('ayy')
# filtered, clean_data, cvs, agg, cvs_2, agg_2 = dc.generate_stats(clean_data, filtered, ['0103', '0102', '0101'])
# cvs_2.sort_values(by=['JobRef', 'StartDateTime'], ascending=True)

#
# _, _, cvs, agg = dc.generate_stats(clean_data=orig_data, filtered=filtered, gates=['0103', '0102', '0101'], inplace=True)
# cvs = cvs.sort_values(by='Start Time')
# new_data = dc.merge_pace_and_non_duplicate(orig_data, filtered, "0102")
# new_data = dc.merge_pace_and_non_duplicate(new_data, filtered, "0103")



# data = dc.filter_data(new_data, filtered)
# data = dc.merge_non_duplicate_deac(data, filtered, 'Indgang 0101')
#
# data.sort_values(by='Date', inplace=True)
# data = data[~data.index.duplicated(keep='first')]
#
#
# a = dc.add_pace_and_non_duplicate(orig_data, filtered, '0102')
# newest_data = pd.merge(orig_data, a, how='left', left_index=True, right_index=True)
# newest_data.drop('Index', axis=1, inplace=True)
# newest_data['Non Duplicate 0102 Counts'].replace(np.nan, 0, inplace=True)
# newest_data['0102 Pace'].replace(np.nan, 0, inplace=True)
#
# for i in range(1, 7):
#     g = 'Indgang 010{}'.format(i)
#     newest_data[g] = newest_data[g].astype(np.int8)
#
# elements = ['Non Duplicate 0102 Counts', '0102 Pace']
# for el in elements:
#     newest_data[el] = newest_data[el].astype(np.int64)
#
# b = dc.add_pace_and_non_duplicate(newest_data, filtered, '0103')
# newest_data = pd.merge(newest_data, b, how='left', left_index=True, right_index=True)
#
# newest_data.drop('Index', axis=1, inplace=True)
# newest_data['Non Duplicate 0103 Counts'].replace(np.nan, 0, inplace=True)
# newest_data['0103 Pace'].replace(np.nan, 0, inplace=True)
#
# for i in range(1, 7):
#     g = 'Indgang 010{}'.format(i)
#     newest_data[g] = newest_data[g].astype(np.int8)
#
# elements = ['Non Duplicate 0103 Counts', '0103 Pace']
# for el in elements:
#     newest_data[el] = newest_data[el].astype(np.int64)


# data = pd.read_csv(r'C:\Users\1067332\Desktop\Dolle\Data\Ladder Machine 1\output.csv', parse_dates=True)
# data.Date = pd.to_datetime(data.Date)
#
# errors = new_data[(new_data['Indgang 0101'] == 0) ]

# f = fragment['Indgang 0103']
# p = f.shift(1)
# fragment['Non Duplicate 0103'] = np.where((f == 1) & (p == 0), 1, 0)
# indices = fragment.index[fragment['Non Duplicate 0103'] != 0][::2]
# fragment['Non Duplicate 0103'] = np.where(fragment.index.isin(indices), 1, 0)
#
# t1 = fragment['Date'][fragment['Non Duplicate 0103'] == 1]
# t2 = t1.shift(1)
# deltas = t1 - t2
# deltas = deltas.dt.seconds
# deltas.replace(np.nan, 0, inplace=True)
# fragment['0103 Pace'] = 0
# fragment['0103 Pace'][fragment['Non Duplicate 0103'] == 1] = deltas
#
# f = fragment['Indgang 0102']
# p = f.shift(1)
# fragment['Non Duplicate 0102'] = np.where((f == 1) & (p == 0), 1, 0)
#
# t1 = fragment['Date'][fragment['Non Duplicate 0102'] == 1]
# t2 = t1.shift(1)
#
# deltas = t1 - t2
# deltas = deltas.dt.seconds
# deltas.replace(np.nan, 0, inplace=True)
# fragment['0102 Pace'] = 0
# fragment['0102 Pace'][fragment['Non Duplicate 0102'] == 1] = deltas
#
# f = fragment['Indgang 0101']
# p = f.shift(1)
# fragment['Non Duplicate 0101'] = np.where((f == 0) & (p == 1) & (f.index != len(f.index) - 1), 1, 0)
#
#
# # f['Non Duplicate 0102'] = np.where((f == 1) & (p == 0), 1, 0)


def pace_since_error_data_0103(o_data, path):
    """
    Whenever gate 0101 flips from 1 to 0, the time since the last input in gate 0102 and the last output from
    gate 0103 are recorded and returned as a dataframe. The key is the startdate of the last non duplicate
    0103 output. Can therefore be merged with tables containing non-duplicate 0103 row data

    :param o_data:
    :param path:
    :return: dataframe based on non duplicate 0103 output
    """
    keys = ['JobRef']
    funcs = (_per_job_generic_calc_data, _per_episode_calc_deac_time_deltas)
    o = _calc_generic_data(o_data, _filtered_worktable(path), funcs, keys)
    return pd.DataFrame(o, columns=['Start', 'End', 'JobRef', 'delta_0102', 'delta_0103'])



def _per_job_generic_calc_data(job_id, f, func):
    indices = f[f['Non-Duplicate 0103'] != 0].copy()
    indices.reset_index(inplace=True)
    f_indices = indices.shift(-1)
    f_indices.fillna(0, inplace=True)
    f_indices['index'] = f_indices['index'].astype(np.int64, inplace=True)
    return func(indices, f_indices, f, job_id)


def _per_episode_calc_deac_time_deltas(indices, f_indices, f, job_id):
    current_0102 = -1
    output = []
    for i in indices.index:
        if i != len(indices.index) - 1:
            new_data = f[(f.index >= indices.loc[i, 'index']) & (f.index <= f_indices.loc[i, 'index'])]

            if isinstance(new_data, pd.DataFrame):
                new_data.reset_index(inplace=True)
                start = new_data.at[0, 'Date']
                end = new_data.iloc[-1, 1]

                current_0102, deac_times = _per_episode_calc_deactivated_times(start, end, job_id, new_data,
                                                                               current_0102)
                output.extend(deac_times)
    return output


def _per_episode_calc_deactivated_times(beginning, end, job_id, data, current_0102):
    f_data = _shift_fillna_int(data, 'Indgang 0102', -1)
    deac_data = []
    chain_started = False
    for i in data.index:
        if data.at[i, 'Indgang 0101'] == 0:
            if f_data.at[i, 'Indgang 0101'] == 1:
                if current_0102 == -1:
                    delta_0102 = 0
                else:
                    delta_0102 = _attempt_pace_subtraction(current_0102, f_data.at[i, 'Date'])

                delta_0103 = _attempt_pace_subtraction(beginning, f_data.at[i, 'Date'])
                deac_data.append([beginning, end, job_id, delta_0102, delta_0103])
                current_0102 = f_data.at[i, 'Date']
                chain_started = False

            elif f_data.at[i, 'Indgang 0101'] == 0 and chain_started is False:
                chain_started = True
                current_0102 = data.at[i, 'Date']

    return current_0102, deac_data


def _per_episode_calc_deactivated_times(beginning, end, job_id, data, current_0102):
    f_data = _shift_fillna_int(data, 'Indgang 0102', -1)
    deac_data = []
    chain_started = False
    for i in data.index:
        if data.at[i, 'Indgang 0101'] == 0:
            if f_data.at[i, 'Indgang 0101'] == 1:
                if current_0102 == -1:
                    delta_0102 = 0
                else:
                    delta_0102 = _attempt_pace_subtraction(current_0102, f_data.at[i, 'Date'])

                delta_0103 = _attempt_pace_subtraction(beginning, f_data.at[i, 'Date'])
                deac_data.append([beginning, end, job_id, delta_0102, delta_0103])
                current_0102 = f_data.at[i, 'Date']
                chain_started = False

            elif f_data.at[i, 'Indgang 0101'] == 0 and chain_started is False:
                chain_started = True
                current_0102 = data.at[i, 'Date']

    return current_0102, deac_data
