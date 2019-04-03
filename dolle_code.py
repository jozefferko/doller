import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

"""
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Functions which are used to load data sheets////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


def clean_csv(path_in, path_out, output, sep=';'):
    """

    :param path_in:
    :param path_out:
    :param output:
    :param sep:
    :return:
    """
    try:
        data = pd.read_csv(path_in, sep)
        if 'Indgang 0101' in data.columns:
            try:
                data = read_ladder_one(path_in)
                data.sort_values(by='Date', inplace=True)
                data.reset_index(inplace=True, drop=True)
                if output:
                    data.to_csv(path_out, sep=sep, index_label=False)
                return data
            except:
                pass
        else:
            pass

    except:
        print("Couldn't read provided path")


def read_ladder_one(path):
    drop_list = [
        'Indgang 0107', 'Indgang 0108', 'Indgang 0109', 'Indgang 0110',
        'Indgang 0111', 'Indgang 0112', 'Indgang 0113', 'Indgang 0114',
        'Indgang 0115', 'Indgang 0116', 'Indgang 0201', 'Indgang 0202',
        'Indgang 0203', 'Indgang 0204', 'Indgang 0205', 'Indgang 0206',
        'Indgang 0207', 'Indgang 0208', 'Indgang 0209', 'Indgang 0210',
        'Indgang 0211', 'Indgang 0212', 'Indgang 0213', 'Indgang 0214',
        'Indgang 0215', 'Indgang 0216', 'Indgang 0301', 'Indgang 0302',
        'Indgang 0303'
    ]
    return _read(path, drop_list, '010')


def _read(path, drop_list, prefix):
    data = pd.read_csv(path, sep=';')
    if len(drop_list) > 0:
        data.drop(drop_list, axis=1, inplace=True)
    return _fix_dates(data, prefix)


# def load_worktable(path):
#     """
#     Reads a cleaned work table from a directory, converts its StartDateTime and StopDateTime columns into panda datetimes
#     then returns a pandas dataframe
#
#     :param path: directory pointing to a work table cav
#     :return:
#     """
#     work_table = pd.read_csv(path, sep=';')
#     work_table['StartDateTime'] = pd.to_datetime(work_table['StartDateTime'])
#     work_table['StopDateTime'] = pd.to_datetime(work_table['StopDateTime'])
#     return work_table


def filter_raw_work_table(path_wrk, path_prod, machine_id, reg_ex):
    prod_table = pd.read_csv(path_prod, sep=';', encoding="ISO-8859-1")
    work_table = pd.read_csv(path_wrk, sep=';')
    work_table['SysQtyGood'] = convert_to_float(work_table['SysQtyGood'])
    work_table['StartDateTime'] = work_table['StartDate'] + ' ' + work_table['StartTime']
    work_table['StopDateTime'] = work_table['StopDate'] + ' ' + work_table['StopTime']
    work_table['StartDateTime'] = work_table['StartDateTime'].apply(lambda x: _split_string_into_dt(x))
    work_table['StopDateTime'] = work_table['StopDateTime'].apply(lambda x: _split_string_into_dt(x))
    work_table = work_table.merge(prod_table[['Name', 'ProdId']], left_on='JobRef', right_on='ProdId')
    filtered = _filter_work_table_by_work_id(work_table, machine_id, reg_ex)
    filtered = filtered.reset_index(drop=True)
    return filtered


def _filter_work_table_by_work_id(sqnce, machine_id, reg_ex):
    """
    Pass in a worktable csv and return only the rows which match the machine ID and the reg_ex which defines what
    product(s) the machine is outputting

    :param sqnce: pandas dataframe
    :param wrk_id:
    :param reg_ex:
    :return:
    """
    return pd.DataFrame(sqnce[(sqnce['WrkCtrId'] == machine_id) &
                              (sqnce['StartDateTime'] >= sqnce.loc[0, ]['StartDateTime']) &
                              (sqnce['Name'].str.contains(reg_ex, regex=True))])



"""
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Functions which are used to modify or add information to data sheets////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


def _generic(func, args, col=None, default=True):
    chunks = []
    data = args[0]
    wrk_table = args[1]
    for i in wrk_table.index:
        fragment = _basic_generic_data(data, i, wrk_table, with_return_vals=False)
        if len(fragment.index) != 0:
            if col is not None:
                fragment = fragment[col]
            if len(args) > 2:
                chunks.append(func(fragment, args[2:]))
            else:
                chunks.append(func(fragment))
    if default:
        return pd.concat(chunks)
    else:
        return chunks


def _basic_generic_data(data, i, wrk_table, with_return_vals=True):
    start = wrk_table.loc[i, 'StartDateTime']
    stop = wrk_table.loc[i, 'StopDateTime']
    fragment = data[(data['Date'] >= start) & (data['Date'] <= stop)]
    fragment = fragment[~fragment.index.duplicated(keep='first')]
    if with_return_vals:
        return start, stop, fragment
    else:
        return fragment


def assemble_data(f):
    fragment['Non Duplicate 0103'] = np.where((f == 1) & (p == 0), 1, 0)
    fragment['Non Duplicate 0103'] = np.where(fragment.index[fragment['Non Duplicate 0103'] != 0][::2], 1, 0)
    fragment['Non Duplicate 0102'] = np.where((f == 1) & (p == 0), 1, 0)


def assemble_data(orig_data, filtered, args):
    new_data = merge_pace_and_non_duplicate(orig_data, filtered, args[0])
    new_data = merge_pace_and_non_duplicate(new_data, filtered, args[1])
    data = filter_data(new_data, filtered)
    data = merge_non_duplicate_deac(data, filtered, 'Indgang {}'.format(args[2]))
    data.sort_values(by='Date', inplace=True)
    data = data[~data.index.duplicated(keep='first')]
    return data


def merge_pace_and_non_duplicate(orig_data, filtered, gate):
    a = _generic(_calc_pace, [orig_data, filtered, gate])
    new_data = pd.merge(orig_data, a, how='left', left_index=True, right_index=True)
    new_data.drop('Index', axis=1, inplace=True)
    non_dup_count = 'Non Duplicate {} Counts'.format(gate)
    gate_pace = '{} Pace'.format(gate)
    new_data[non_dup_count].replace(np.nan, 0, inplace=True)
    new_data[gate_pace].replace(np.nan, 0, inplace=True)

    if gate[:3] == '010':
        str_base = 'Indgang 010'
    else:
        str_base = ''
    for i in range(1, 7):
        g = '{0}{1}'.format(str_base, i)
        new_data[g] = new_data[g].astype(np.int8)
    elements = [non_dup_count, gate_pace]
    for el in elements:
        new_data[el] = new_data[el].astype(np.int64)
    return new_data


def _calc_pace(fragment, args):
    gate = args[0]
    f = fragment.copy()
    non_duplicate = 'Non Duplicate {} Counts'.format(gate)
    f[non_duplicate] = _compute_count_only(f, 'Indgang {}'.format(gate))
    if gate == '0103':
        indices = f.index[f[non_duplicate] != 0][::2]
    else:
        indices = f.index[f[non_duplicate] != 0]

    df = _create_indexed_column('col1', indices, 'col2', np.ones(indices.size).astype(np.int64))
    f = _left_merge(f, df, 'col2')

    deltas = _create_shifted_times(f['Date'][f['col2'] != 0])
    df = _create_indexed_column('col3', indices, 'col4', deltas)
    f = _left_merge(f, df, 'col4')
    return pd.DataFrame({'Index': f.index, non_duplicate: f['col2'], '{} Pace'.format(gate): f['col4']})


def merge_non_duplicate_deac(orig_data, filtered, gate):
    a = _generic(_non_duplicate_deac, [orig_data, filtered, gate])
    a.set_index('Index', inplace=True)
    new_data = pd.merge(orig_data, a, how='left', left_index=True, right_index=True)
    new_data['Non Duplicate Deactivated'].replace(np.nan, 0, inplace=True)
    return new_data


def _non_duplicate_deac(fragment, args):
    present = fragment.copy()
    present.reset_index(inplace=True)
    last = present.tail(1)
    gate = args[0]
    if int(last[gate]) == 1:
        num = 1
    else:
        num = 0
    future = _shift_fillna_int(present, -1, num)
    chunks = []
    for i in present.index:
        if i != len(present.index)-1:
            if present.at[i, gate] == 0 and future.at[i, gate] == 1:
                chunks.append([present.at[i, 'index'], 1])
    return pd.DataFrame(chunks, columns=['Index', 'Non Duplicate Deactivated'])


"""
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Functions which are used to generate statistics about the data//////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


def generate_stats(clean_data, filtered, args):
    cols = [
        ['Job ID', 'Product', 'StartDateTime', 'StopDateTime', '{} Sum'.format(args[0]), 'Expected {} Sum'.format(args[0])],
        ['StartDateTime', 'StopDateTime', '{} Sum'.format(args[1]), 'Expected {} Sum'.format(args[1])]
    ]
    cvs, agg = _gen_cvs_aggs(clean_data, filtered, args, cols)
    no_overlaps = pd.DataFrame(_merge_overlaps(cvs), columns=[
        'StartDateTime', 'StopDateTime', 'SysQtyGood', 'JobRef', 'Name'])
    no_overlaps = no_overlaps.sort_values('StartDateTime')
    cvs_2, agg_2 = _gen_cvs_aggs(clean_data, no_overlaps, args, cols)
    return filtered, clean_data, cvs, agg, cvs_2, agg_2


def _gen_cvs_aggs(clean_data, filtered, gates, cols):
    cvs = _gen_cvs(clean_data, filtered, gates[0], cols[0])
    cvs = _merge_cvs(clean_data, filtered, cvs, gates[1], cols[1])
    cvs.sort_values(by=['Job ID', 'StartDateTime'], ascending=True)
    agg = aggregated_count_vs_expected(cvs, gates)
    return cvs, agg


def _merge_cvs(clean_data, filtered, cvs, gate, cols):
    cvs_2 = _gen_cvs(clean_data, filtered, gate, cols, reduced=True)
    cvs_2 = cvs_2.iloc[:, 2:]
    cvs = pd.merge(cvs, cvs_2, how='left', left_index=True, right_index=True)
    deacs = _gen_deacs(args=[clean_data, cvs])
    cvs['0101 Sum'] = deacs
    columns = ['Job ID', 'StartDateTime', 'StopDateTime',
               '0101 Sum', '0102 Sum', 'Expected 0102 Sum',
               '0103 Sum', 'Expected 0103 Sum', 'Product']
    return pd.DataFrame(cvs, columns=columns)


def _gen_cvs(clean_data, filtered, gate, cols, reduced=False):
    cvs = _count_vs_expected(clean_data, filtered, gate, cols, reduced=reduced)
    cvs = cvs.sort_values(by='StartDateTime')
    cvs = cvs[~cvs[cols].duplicated(keep='first')]
    cvs = cvs.reset_index(drop=True)
    return cvs


def _gen_deacs(args):
    return _generic(_pd_counts_with_duplicates, args, col='Non Duplicate Deactivated', default=False)


def _count_vs_expected(data, filtered, gate, cols, path_out_output=None, reduced=False):
    output_data = pd.DataFrame(_create_count_vs_expected_statistics(data, filtered, gate, reduced=reduced), columns=cols)
    if path_out_output is not None:
        output_data.to_csv(path_out_output, sep=';')
    return output_data


def _create_count_vs_expected_statistics(data, wrk_table, gate, reduced=False):
    temp = []
    for i in wrk_table.index:
        start, stop, fragment = _basic_generic_data(data, i, wrk_table)
        count = _compute_count_with_duplicates(fragment['Indgang {}'.format(gate)])
        fragment['Count'] = count
        times = _create_shifted_times(fragment['Date'][fragment['Count'] != 0])
        count = np.sum(_compute_count_with_times(fragment, times, 'Count'))
        qty = wrk_table.at[i, 'SysQtyGood'].astype(np.int32)
        if len(fragment.index) != 0:
            if reduced:
                qty *= 6
                output = [
                    start, stop, count, qty
                ]
            else:
                count //= 2
                output = [
                    wrk_table.at[i, 'JobRef'], wrk_table.at[i, 'Name'], start, stop, count, qty
                ]
            temp.append(output)
    return temp


def _gen_quantities(f, i):
    _, p = _create_shifted_lists(f, 0)
    p_stop = p.at[i, 'StopDateTime']
    start = f.at[i, 'StartDateTime']
    stop = f.at[i, 'StopDateTime']
    qty = f.at[i, 'Expected 0103 Sum']
    job = f.at[i, 'Job ID']
    return p_stop, start, stop, qty, job


def _merge_overlaps(cvs):
    job_list = cvs[~cvs['Job ID'].duplicated(keep='first')]
    chunks = []
    temp = []
    partial_overlap = []
    for i in job_list.index:
        f = cvs[cvs['Job ID'] == job_list.at[i, 'Job ID']]
        f = f.reset_index(drop=True)
        for i in f.index:
            p_stop, start, stop, qty, job = _gen_quantities(f, i)
            product = f.at[i, 'Product']
            if i == 0:
                temp.append([start, stop, qty, job, product])
            else:
                if start < p_stop < stop:
                    if len(partial_overlap) == 0 and len(temp) == 1:
                        partial_overlap.append(temp.pop())
                    partial_overlap[0][1] = stop
                    partial_overlap[0][2] += qty

                elif p_stop <= start and p_stop < stop:
                    if len(partial_overlap) == 1 and len(temp) == 0:
                        chunks.append(partial_overlap.pop())
                    if len(partial_overlap) == 0 and len(temp) == 1:
                        chunks.append(temp.pop())
                    temp.append([start, stop, qty, job, product])

        if len(partial_overlap) == 1 and len(temp) == 0:
            chunks.append(partial_overlap.pop())
        if len(partial_overlap) == 0 and len(temp) == 1:
            chunks.append(temp.pop())
    return chunks


def aggregated_count_vs_expected(stats, agg_cols, path_out=None):
    unique = stats.drop_duplicates(subset=['Job ID'])
    grouped = _calc_ratio(stats, agg_cols[0])
    grouped = _calc_ratio(grouped, agg_cols[1])

    grouped.loc[:, 'Product'] = np.array(unique['Product'])
    grouped['StartDateTime'] = _add_times(stats, 'StartDateTime')
    grouped['StopDateTime'] = _add_times(stats, 'StopDateTime')
    grouped = grouped.sort_values(by='StartDateTime')
    grouped.reset_index(inplace=True)

    columns = ['Job ID', 'StartDateTime', 'StopDateTime', '0101 Sum', '0102 Sum', 'Expected 0102 Sum', '0102 Ratio',
               '0103 Sum', 'Expected 0103 Sum', '0103 Ratio', 'Product']
    grouped = pd.DataFrame(grouped, columns=columns)
    if path_out is not None:
        grouped.to_csv(path_out, sep=';')
    return grouped


def _calc_ratio(stats, agg_col):
    ratio = '{} Ratio'.format(agg_col)
    expected = 'Expected {} Sum'.format(agg_col)
    grouped = stats.groupby(['Job ID']).sum()
    grouped[expected] = grouped[expected].astype(np.int32)
    grouped[ratio] = grouped['{} Sum'.format(agg_col)].astype('f') / grouped[expected]
    return grouped


def _add_times(stats, col):
    time = stats[[col, 'Job ID']].groupby('Job ID')
    if col == 'StartDateTime':
        return time[col].min()
    else:
        return time[col].max()


def _compute_count_only(f, col):
    f, p = _create_shifted_lists(f[col], 0)
    return [1 if x else 0 for x in (f == 1) & (p == 0)]


def _compute_count_with_times(f, t, col):
    f, p = _create_shifted_lists(f[col], 0)
    return [1 if x else 0 for x in (f == 1) & (p == 0) | (f == 1) & (t > 25)]


def _create_shifted_times(times):
    t1 = times.copy()
    t2 = t1.shift(1)
    times = t1 - t2
    times = times.dt.seconds
    return times


def _left_merge(f, df, col):
    f = f.merge(df, how='left', left_index=True, right_index=True)
    f[col].fillna(0, inplace=True)
    f[col] = f[col].astype(np.int64)
    return f


def _create_indexed_column(col_1_name, col_1_value, col_2_name, col_2_value):
    d = {col_1_name: col_1_value, col_2_name: col_2_value}
    df = pd.DataFrame(d)
    df.set_index(col_1_name, drop=False, inplace=True)
    return df


def filter_data(data, wrk_table):
    chunks = []
    for i in wrk_table.index:
        start = wrk_table.at[i, 'StartDateTime']
        stop = wrk_table.at[i, 'StopDateTime']
        fragment = data[(data['Date'] >= start) & (data['Date'] <= stop)]
        fragment['JobRef'] = wrk_table.at[i, 'JobRef']
        chunks.append(fragment)
    return pd.concat(chunks)


def _compute_count_only(f, col):
    f, p = _create_shifted_lists(f[col], 0)
    return [1 if x else 0 for x in (f == 1) & (p == 0)]


def compute_count_only_with_delta(f, col, deltas):
    f, p = _create_shifted_lists(f[col], 0)
    return [[x.index, 1] if x else 0 for x in (f == 1) & (p == 0)]


def _pd_counts_with_duplicates(f):
    return np.sum(_compute_count_with_duplicates(f))


def _compute_count_with_duplicates(f):
    return [1 if x else 0 for x in f == 1]


def _create_shifted_lists(fragment, y):
    f = fragment
    p = f.shift(1)
    p.fillna(y, inplace=True)
    return f, p


def _gen_dates(date, time):
    date = str(date) + ' ' + str(time)
    return date


def _fix_dates(d, prefix):
    d['Time'] = d['Time'].astype(str)
    for i in d.index:
        if d.at[i, 'Time'] != '0':
            d.at[i, 'Date'] = _gen_dates(d.at[i, 'Date'], d.at[i, 'Time'])
    d.drop('Time', axis=1, inplace=True)
    d['Date'] = pd.to_datetime(d['Date'])
    for i in range(1, 7):
        d['Indgang {0}{1}'.format(prefix, i)].replace(np.nan, 0, inplace=True)
        d['Indgang {0}{1}'.format(prefix, i)] = d['Indgang {0}{1}'.format(prefix, i)].astype(np.int8)
    return d


def _split_string_into_dt(date):
    d = re.split("[-:/. ]", date)
    return pd.datetime(int(d[2]), int(d[1]), int(d[0]), int(d[3]), int(d[4]), int(d[5]))




"""
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
"""


def add_accumulated_0103(data, wrk_table):
    chunks = []
    for i in wrk_table.index:
        start = wrk_table.at[i, 'StartDateTime']
        stop = wrk_table.at[i, 'StopDateTime']
        fragment = data[(data['Date'] >= start) & (data['Date'] <= stop)]
        fragment = fragment[~fragment.index.duplicated(keep='first')]
        chunks.append(_calc_pace(fragment, gate))
    return pd.concat(chunks)


def add_accumulated_0103_data(indices, f_indices, f, job_id):
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


def add_pace_and_non_duplicate(data, wrk_table, gate):
    chunks = []
    for i in wrk_table.index:
        start = wrk_table.at[i, 'StartDateTime']
        stop = wrk_table.at[i, 'StopDateTime']
        fragment = data[(data['Date'] >= start) & (data['Date'] <= stop)]
        fragment = fragment[~fragment.index.duplicated(keep='first')]
        chunks.append(_calc_pace(fragment, gate))
    return pd.concat(chunks)


def _calc_generic_data(data, wrk_table, funcs, keys):
    """
    This is a generic function called by top level, data aggregation, user functions. It takes a tuple of functions
    and keys and using these values calls lower level non generic functions

    :param data: dataframe of ladder machine data
    :param wrk_table:
    :param funcs: a tuple of function names
    :param keys: a list of strings
    :return:
    """
    temp = []
    for i in wrk_table.index:
        start = wrk_table.at[i, 'StartDateTime']
        stop = wrk_table.at[i, 'StopDateTime']
        fragment = data[(data['Date'] > start) & (data['Date'] < stop)]
        if len(fragment.index != 0):
            if len(keys) == 0 and len(funcs) == 1:
                temp.extend(funcs[1][0](fragment))

            if 'JobRef' in keys and len(keys) == 1:
                job = wrk_table.at[i, 'JobRef']
                if len(funcs) == 1:
                    temp.extend(funcs[0](job, fragment))
                if len(funcs) == 2:
                    temp.extend(funcs[0](job, fragment, funcs[1]))

            else:
                if len(funcs) == 1:
                    temp.extend(funcs[0](fragment))

    return temp


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
    return pd.DataFrame(o, columns=['Start', 'End', 'Job ID', 'delta_0102', 'delta_0103'])


def per_episode_calc_error_data(data, col, off=False):
    f_data = _shift_fillna_int(data, -1, 0, col)
    p_data = _shift_fillna_int(data, 1, 0, col)

    start = None
    times = list()
    deactivated_times = list()
    count = 0

    for i in data.index:
        if p_data.at[i, col] == 0 and data.at[i, col] == 1 and f_data.at[i, col] == 1:
            start = data.loc[i, :]

        if p_data.at[i, col] == 0 and data.at[i, col] == 1 and f_data.at[i, col] == 0:
            times.append([_attempt_time_calc(data.at[i, 'Date'], f_data.at[i, 'Date']), 1, True])
            count += 1

        if p_data.at[i, col] == 1 and data.at[i, col] == 1 and f_data.at[i, col] == 0:
            times.append([_attempt_time_calc(start['Date'], data.at[i, 'Date']), 1, False])
            count += 1
            start = None

        if off:
            if data.at[i, 'Indgang 0101'] == 0 and f_data.at[i, 'Indgang 0101'] == 1:
                deactivated_times.append(_attempt_time_calc(data.at[i, 'Date'], f_data.at[i, 'Date']))

    names = ['{} Duration'.format(col[-4:]), 'Count', 'Single Error']

    if len(times) == 1:
        errors = pd.Series(times[0], index=names)
    else:
        errors = pd.DataFrame(times, columns=names)

    if off:
        total_off_time = sum(deactivated_times)
        return errors, total_off_time
    else:
        return errors


def _shift_fillna_int(sqnce, pos, error, col=None):
    sqnce = sqnce.shift(pos, axis=0)
    sqnce.fillna(error, inplace=True)
    if col is not None:
        sqnce[col] = sqnce[col].astype(int)
    return sqnce


def _attempt_pace_subtraction(start, finish):
    try:
        time = finish - start
        time = time.seconds
        return time
    except:
        return 0


def convert_to_float(d):
    d = d.str.split(',')
    d = d.str.join('.')
    return d.astype('float')

