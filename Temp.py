import pandas as pd
import numpy as np


def _generic(func, args):
    chunks = []
    data = args[0]
    wrk_table = args[1]
    for i in wrk_table.index:
        fragment = _basic_generic_data(data, i, wrk_table)
        if len(fragment.index) != 0:
            chunks.append(func(fragment, args[2:]))
    return pd.concat(chunks)


def _basic_generic_data(data, i, wrk_table, default=True):
    start = wrk_table.loc[i, 'StartDateTime']
    stop = wrk_table.loc[i, 'StopDateTime']
    fragment = data[(data['Date'] >= start) & (data['Date'] <= stop)]
    fragment = fragment[~fragment.index.duplicated(keep='first')]
    if default:
        return fragment
    else:
        return start, stop, fragment


def assemble_data(data, work_table, gates, columns):
    return pd.DataFrame(_generic(_assemble_gate_counts, [data, work_table, gates]), columns=columns)


def _assemble_gate_counts(fragment, gates):
    gates = gates[0]
    fragment['{} Pace'.format(gates[1])] = 0
    fragment['{} Pace'.format(gates[2])] = 0
    fragment['Non Duplicate {}'.format(gates[0])] = _add_deac_counts(fragment, gates[0])
    fragment['Non Duplicate {}'.format(gates[1])] = _add_gate_counts(fragment, gates[1])
    fragment['{} Pace'.format(gates[1])][fragment['Non Duplicate {}'.format(gates[1])] == 1] = _add_gate_pace(fragment, gates[1])
    fragment['Non Duplicate {}'.format(gates[2])] = _add_gate_counts(fragment, gates[2], reduced=2)
    fragment['{} Pace'.format(gates[2])][fragment['Non Duplicate {}'.format(gates[2])] == 1] = _add_gate_pace(fragment, gates[2])
    return fragment


def _add_gate_counts(fragment, gate, reduced=None):
    f, p = _return_f_and_p(fragment, gate)
    if reduced is not None:
        fragment['Non Duplicate {}'.format(gate)] = np.where((f == 1) & (p == 0), 1, 0)
        indices = fragment.index[fragment['Non Duplicate {}'.format(gate)] != 0][::reduced]
        return np.where(fragment.index.isin(indices), 1, 0)
    return np.where((f == 1) & (p == 0), 1, 0)


def _add_gate_pace(fragment, gate):
    fragment = fragment.copy()
    t1 = fragment['Date'][fragment['Non Duplicate {}'.format(gate)] == 1]
    deltas = t1.sub(t1.shift(1)).dt.seconds
    deltas.replace(np.nan, 0, inplace=True)
    return deltas


def _add_deac_counts(fragment, gate):
    f, p = _return_f_and_p(fragment, gate)
    return np.where((f == 0) & (p == 1) & (f.index != len(f.index) - 1), 1, 0)


def _return_f_and_p(fragment, gate):
    f = fragment['Indgang {}'.format(gate)]
    p = f.shift(1)
    return f, p


def gen_statistics(data, wrk_table, gates, args):
    cvs, agg = _cvs_and_agg(data, wrk_table, gates, args[1:])
    # no_duplicates = pd.DataFrame(_remove_partial_overlaps(cvs), columns=args[0])
    # no_duplicates.sort_values('StartDateTime', inplace=True)
    # cvs_2, agg_2 = _cvs_and_agg(data, no_duplicates, gates, args[1:])
    return cvs, agg


def _cvs_and_agg(data, wrk_table, gates, args):
    columns_1 = args[0]
    columns_2 = args[1]
    agg_cols = gates[1:]
    cvs = pd.DataFrame(_create_count_vs_expected_statistics(data, wrk_table, gates), columns=columns_1)
    cvs['Median {} Pace'.format(gates[1])].replace(np.nan, 0, inplace=True)
    cvs['Median {} Pace'.format(gates[2])].replace(np.nan, 0, inplace=True)
    cvs.sort_values(by=['JobRef', 'StartDateTime'], ascending=[True, True], inplace=True)
    agg = aggregated_count_vs_expected(cvs, agg_cols, columns_2)
    agg.sort_values(by='JobRef', ascending=True, inplace=True)
    agg.apply(lambda x: x.replace(np.nan, 0, inplace=True))
    agg.reset_index(drop=True, inplace=True)
    return cvs, agg


def _create_count_vs_expected_statistics(data, wrk_table, gates):
    temp = []
    for i in wrk_table.index:
        start, stop, fragment = _basic_generic_data(data, i, wrk_table, default=False)
        if len(fragment.index) != 0:
            count_1 = np.sum(fragment['Non Duplicate {}'.format(gates[0])])
            count_2 = np.sum(fragment['Non Duplicate {}'.format(gates[1])])
            count_3 = np.sum(fragment['Non Duplicate {}'.format(gates[2])])

            pace_2 = '{} Pace'.format(gates[1])
            no_zeroes_2 = fragment[pace_2][fragment[pace_2] != 0]
            median_pace_2 = np.median(no_zeroes_2)
            mean_pace_2 = np.mean(fragment[pace_2])

            pace_3 = '{} Pace'.format(gates[2])
            no_zeroes_3 = fragment[pace_3][fragment[pace_3] != 0]
            median_pace_3 = np.median(no_zeroes_3)
            mean_pace_3 = np.mean(no_zeroes_3)

            init_median_pace_2 = 0
            init_mean_pace_2 = 0
            no_zeroes_in = fragment['Non Duplicate {}'.format(gates[1])][fragment['Non Duplicate {}'.format(gates[1])]!=0]
            if not no_zeroes_in.empty:
                corrected_slice = fragment.loc[no_zeroes_in.index[0]:]
                no_zeroes_out = corrected_slice['Non Duplicate {}'.format(gates[2])][fragment['Non Duplicate {}'.format(gates[2])]!=0]
                if not no_zeroes_out.empty:
                    #init_pace = fragment.loc[no_zeroes_in.index[0]]['Date']-fragment.loc[no_zeroes_out.index[0]]['Date']
                    corrected_slice = corrected_slice.loc[:no_zeroes_out.index[0]]
                    init_no_zeroes_2 = corrected_slice['{} Pace'.format(gates[1])][corrected_slice['{} Pace'.format(gates[1])] != 0]
                    init_median_pace_2 = np.median(init_no_zeroes_2)
                    init_mean_pace_2 = np.mean(init_no_zeroes_2)
            qty = wrk_table.at[i, 'SysQtyGood'].astype(np.int32)
            output = [
                wrk_table.at[i, 'JobRef'],
                start,
                stop,
                count_1,
                count_2, qty*6,init_median_pace_2, median_pace_2, init_mean_pace_2,mean_pace_2,
                count_3, qty//2, median_pace_3, mean_pace_3,
                wrk_table.at[i, 'Name']
            ]
            temp.append(output)
    return temp


def aggregated_count_vs_expected(stats, agg_cols, columns_agg):
    grouped = _calc_ratio(stats, agg_cols[0])
    grouped = _calc_ratio(grouped, agg_cols[1])

    median_1 = 'Median {} Pace'.format(agg_cols[0])
    mean_1 = 'Mean {} Pace'.format(agg_cols[0])
    median_2 = 'Median {} Pace'.format(agg_cols[1])
    mean_2 = 'Mean {} Pace'.format(agg_cols[1])

    grouped[median_1] = _calc_averages(stats, median_1)
    grouped[mean_1] = _calc_averages(stats, mean_1)
    grouped[median_2] = _calc_averages(stats, median_2)
    grouped[mean_2] = _calc_averages(stats, mean_2)

    unique = stats.drop_duplicates(subset=['JobRef'])
    grouped.loc[:, 'Product'] = np.array(unique['Product'])
    grouped['StartDateTime'] = _add_times(stats, 'StartDateTime')
    grouped['StopDateTime'] = _add_times(stats, 'StopDateTime')
    grouped = grouped.sort_values(by='StartDateTime')
    grouped.reset_index(inplace=True)
    grouped = pd.DataFrame(grouped)
    return grouped


def _calc_averages(stats, column):
    stats['Temp'] = np.where(stats[column] > 0, 1, 0)
    return stats.groupby('JobRef').apply(lambda x: x[column].sum()/np.sum(x['Temp']))


def _calc_ratio(stats, agg_col):
    ratio = '{} Ratio'.format(agg_col)
    expected = 'Expected {} Sum'.format(agg_col)
    grouped_sum = stats.groupby(['JobRef']).sum()
    grouped_sum[expected] = grouped_sum[expected].astype(np.int32)
    grouped_sum[ratio] = grouped_sum['{} Sum'.format(agg_col)].astype('f') / grouped_sum[expected]
    return grouped_sum


def _add_times(stats, col):
    time = stats[[col, 'JobRef']].groupby('JobRef')
    if col == 'StartDateTime':
        return time[col].min()
    else:
        return time[col].max()


def _create_shifted_lists(fragment, y):
    f = fragment
    p = f.shift(1)
    p.fillna(y, inplace=True)
    return f, p


def _remove_partial_overlaps(cvs):
    job_list = cvs[~cvs['JobRef'].duplicated(keep='first')]
    chunks = []
    temp = []
    partial_overlap = []
    for i in job_list.index:
        f = cvs[cvs['JobRef'] == job_list.at[i, 'JobRef']]
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


def _gen_quantities(f, i):
    _, p = _create_shifted_lists(f, 0)
    p_stop = p.at[i, 'StopDateTime']
    start = f.at[i, 'StartDateTime']
    stop = f.at[i, 'StopDateTime']
    qty = f.at[i, 'Expected 0103 Sum']
    job = f.at[i, 'JobRef']
    return p_stop, start, stop, qty, job