import pandas as pd
import numpy as np


def event_triggered_response(df, parameter, event_times, time_key=None, t_before=10, t_after=10, sampling_rate=60, output_format='tidy'):
    '''
    build event triggered response around a given set of events
    required inputs:
      df: dataframe of input data
      parameter: column of input dataframe to extract around events
      event_times: times of events of interest
    optional inputs:
      time_key: key to use for time (if None (default), will search for either 't' or 'time'. if 'index', use indices)
      t_before: time before each of event of interest
      t_after: time after each event of interest
      sampling_rate: desired sampling rate of output (input data will be interpolated)
      output_format: 'wide' or 'tidy' (default = 'tidy')
    output:
      if output_format == 'wide':
        dataframe with one time column ('t') and one column of data for each event
      if output_format == 'tidy':
        dataframe with columns representing:
            time
            output value
            event number
            event time
    An example use case, recover a sinousoid from noise:
        (also see https://gist.github.com/dougollerenshaw/628c63375cc68f869a28933bd5e2cbe5)
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        # generate some sample data¶
        # a sinousoid corrupted by noise
        # make a time vector
        # go from -10 to 110 so we have a 10 second overage if we analyze from 0 to 100 seconds
        t = np.arange(-10,110,0.001)
        # now build a dataframe
        df = pd.DataFrame({
            'time': t,
            'noisy_sinusoid': np.sin(2*np.pi*t) + np.random.randn(len(t))*3
        })
        # use the event_triggered_response function to get a tidy dataframe of the signal around every event
        # events will simply be generated as every 1 second interval starting at 0.5, since our period here is 1
        etr = event_triggered_response(
            df,
            parameter = 'noisy_sinusoid',
            event_times = [C+0.5 for C in range(0,99)],
            t_before = 1,
            t_after = 1,
            sampling_rate = 100
        )
        # use seaborn to view the result
        # We're able to recover the sinusoid through averaging
        fig, ax = plt.subplots()
        sns.lineplot(
            data = etr,
            x='time',
            y='noisy_sinusoid',
            ax=ax
        )
    '''
    if time_key is None:
        if 't' in df.columns:
            time_key = 't'
        elif 'timestamps' in df.columns:
            time_key = 'timestamps'
        else:
            time_key = 'time'

    _d = {'time': np.arange(-t_before, t_after, 1 / sampling_rate)}
    for ii, event_time in enumerate(np.array(event_times)):

        if time_key == 'index':
            df_local = df.loc[(event_time - t_before):(event_time + t_after)]
            t = df_local.index.values - event_time
        else:
            df_local = df.query(
                "{0} > (@event_time - @t_before) and {0} < (@event_time + @t_after)".format(time_key))
            t = df_local[time_key] - event_time
        y = df_local[parameter]

        _d.update({'event_{}_t={}'.format(ii, event_time)
                  : np.interp(_d['time'], t, y)})
    if output_format == 'wide':
        return pd.DataFrame(_d)
    elif output_format == 'tidy':
        df = pd.DataFrame(_d)
        melted = df.melt(id_vars='time')
        melted['event_number'] = melted['variable'].map(
            lambda s: s.split('event_')[1].split('_')[0])
        melted['event_time'] = melted['variable'].map(
            lambda s: s.split('t=')[1])
        return melted.drop(columns=['variable']).rename(columns={'value': parameter})
