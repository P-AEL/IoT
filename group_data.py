import pandas as pd

def group_data(df, freq):

    # changing date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    #onehot encoding the gateway column and changing its type to int
    df = pd.get_dummies(df, columns=['gateway'])
    for col in ['gateway_drag-lps8-01', 'gateway_drag-lps8-02', 'gateway_drag-lps8-03', 'gateway_drag-lps8-05', 'gateway_drag-lps8-07', 'gateway_drag-lps8-08', 'gateway_drag-outd-01']:
        df[col] = df[col].astype(int)

    #splitting our columns into float and int types
    all_columns_float_type = df.select_dtypes(include=['float64']).columns
    all_columns_float_type = all_columns_float_type.tolist()
    all_columns_float_type.append('device_id')
    all_columns_float_type.append('date_time')
    all_columns_int_type = df.select_dtypes(include=['int']).columns   
    all_columns_int_type = all_columns_int_type.tolist()
    all_columns_int_type.append('device_id')
    all_columns_int_type.append('date_time')

    # grouping by device_id and date_time and aggregating the values separately for float and int columns
    df_hourly_float_values = df[all_columns_float_type].groupby(['device_id', pd.Grouper(key='date_time', freq=freq)]).agg("mean").reset_index()
    df_hourly_int_values = df[all_columns_int_type].groupby(['device_id', pd.Grouper(key='date_time', freq=freq)]).agg("max").reset_index()

    # merging the two dataframes and outputting the result
    merged_df = df_hourly_float_values.merge(df_hourly_int_values, on=['device_id', 'date_time'])
    return merged_df


def group_data_2(df, freq):

    # changing date_time to datetime
    df['date_time'] = pd.to_datetime(df['date_time'])

    #onehot encoding the gateway column and changing its type to int
    df = pd.get_dummies(df, columns=['gateway'])
    for col in ['gateway_drag-lps8-01', 'gateway_drag-lps8-02', 'gateway_drag-lps8-03', 'gateway_drag-lps8-05', 'gateway_drag-lps8-07', 'gateway_drag-lps8-08', 'gateway_drag-outd-01']:
        df[col] = df[col].astype(int)

    all_columns_float_type = df.select_dtypes(include=['float64']).columns

    df_hourly_float_values = df[all_columns_float_type].groupby(['device_id', pd.Grouper(key='date_time', freq=freq)]).agg("mean").reset_index()

    return df