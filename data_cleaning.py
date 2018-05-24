import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate


def preprocess_data(
	fname='data/raw_lab_data.csv',
    droprows=slice(0, 0),
    Xref=640*[0], 
	Xcols=slice(1, 641),
    tcol=0, 
	ycols=slice(641, None),
    nrolling=20, k=3, s=70,
    figname=''
):
    df = pd.read_csv(fname)
    X_labels = df.columns[Xcols]
    signal_labels = df.columns[ycols]

    # drop rows:
    all_data = remove_rows(droprows, df.values)

    # split data from reference:
    data, ref = split_data_ref(all_data, Xref, Xcols)

    tdata, Xdata, yydata = data[:, tcol], data[:, Xcols], data[:, ycols]
    tref, Xref, yyref = ref[:, tcol], ref[:, Xcols], ref[:, ycols]

    df_out = pd.DataFrame(data=Xdata, columns=X_labels, index=tdata)
    df_out.index.name = 'timestamp'

    for ydata, yref, label in zip(yydata.T.copy(), yyref.T.copy(),
                                  signal_labels):
        # some of these steps could have been easier with pandas
        # clean up outliers:
        indxs = find_outliers(yref, tref, nrolling)
        tr, xr, yr = remove_rows(indxs, tref, Xref, yref)

        indxs = find_outliers(ydata, tdata, nrolling)
        td, xd, yd = remove_rows(indxs, tdata, Xdata, ydata)

        # remove time dependence from yd:
        interp_ref = spline_interp(td, tr, yr, k=k, s=s)
        indxs = np.argwhere( interp_ref == 0 )
        td, xd, yd, interp_ref =\
            remove_rows(indxs, td, xd, yd, interp_ref)
        yd /= interp_ref

        # load the data to a pandas dataframe:
        df_sig = pd.DataFrame(index=td)
        # df_sig = pd.DataFrame(data=xd, index=td, columns=X_labels)
        df_sig.index.name = 'timestamp'
        df_sig[label] = yd
        df_sig[label + ' reference'] = interp_ref#;print df_sig.head()

        # join the output dataframe:
        df_out = pd.concat([df_out, df_sig], axis=1)

    if figname:
        plot_preprocessed_data(df_out, figname)

    return df_out


def plot_preprocessed_data(data_out=pd.DataFrame(),
                           figname='preprocessed data'):

    signal_labels = get_signal_labels_from_df(data_out)

    nkeys = len(signal_labels)
    plt.figure(figname, figsize=(10, 2 * nkeys))
    k = 1

    for name in signal_labels:
        ax1 = plt.subplot(nkeys, 1, k)
        ax2 = ax1.twinx()

        data_out.reset_index().plot.scatter(x='timestamp',
                                            y=name, ax=ax1)
        data_out.reset_index().plot.line(x='timestamp',
                                         y=name + ' reference',
                                         ax=ax2,
                                         legend=False)
        plt.ylabel('reference')

        k += 1

    plt.tight_layout()


def get_signal_labels_from_df(df=pd.DataFrame()):
    signal_labels = [name for name in df.columns\
                     if 'pixel' not in name and\
                     'reference' not in name and\
                     'timestamp' not in name]
    return signal_labels


def get_X_from_df(df=pd.DataFrame()):
    cols = ['pixel_' + str(k) for k in range(640)]
    return df[cols].values


def get_signal_subdf(df=pd.DataFrame(),
                     signal_label='SHG'):
    cols = ['timestamp']
    cols += ['pixel_' + str(k) for k in range(640)]
    cols += [signal_label]

    return df[cols]


def join_signal_subdfs(df1=pd.DataFrame(), df2=pd.DataFrame()):
    cols = ['timestamp']
    cols += ['pixel_' + str(k) for k in range(640)]

    merged_df = pd.merge(df1, df2, on=cols, how='inner')

    return merged_df.dropna()


def split_data_ref(all_data=np.array([[]]),
                   Xref=640*[0],
                   Xcols=slice(1, 641)):
    x = np.array(all_data)[:, Xcols]
    ref_indxs = np.all( x - np.array(Xref) == 0, axis=1 )
    data_indxs = np.logical_not( ref_indxs )

    data = np.array(all_data)[data_indxs, :].copy()
    ref = np.array(all_data)[ref_indxs, :].copy()

    return data, ref


def find_outliers(yt, t=[],
                  nrolling=20, label='y',
                  graph=False):
    # if t is given sort it and yt so that the outliers
    # are obtained in the context of neighbors in time.
    if t is []:
        tsorted = np.linspace(0, 1, np.size(y))
        ysorted = np.array(yt).copy()
        sort_indxs = np.arange(np.size(y))
    else:
        sort_indxs = np.argsort(t)
        tsorted = np.array(t)[sort_indxs]
        ysorted = np.array(yt)[sort_indxs]

    m = pd.rolling_mean(ysorted, nrolling)
    s = pd.rolling_std(ysorted, nrolling)

    # finding the outliers:
    outlier_indxs = np.argwhere(np.abs(ysorted - m) > 2.5 * s)
    not_outlier_indxs = np.argwhere(np.abs(ysorted - m) < 2.5 * s)

    if graph:
        plt.figure('outliers removal')
        plt.title('outliers removal')
        plt.plot(tsorted, ysorted, '.')
        plt.xlabel('time')
        plt.ylabel(label)

        plt.plot(tsorted, m, linewidth=4)

        plt.plot(tsorted, m + 2.5 * s, 'g-')
        plt.plot(tsorted, m - 2.5 * s, 'g-')

        # mark identified outliers with an x:
        plt.plot(tsorted[outlier_indxs], ysorted[outlier_indxs], 'rx')

    return sort_indxs[outlier_indxs].ravel()


def spline_interp(xnew, xold, yold,
                  k=3, s=70,
                  graph=False,
                  ylabel='y', xlabel='x'):
    indxs = np.argsort(xold.ravel()).ravel()
    yck = interpolate.splrep(xold.ravel()[indxs], yold.ravel()[indxs],
                             k=k, s=s)
    ynew = interpolate.splev(xnew, yck, ext=3).ravel()

    if graph:
        plt.figure('spline interpolation')
        plt.title('spline interpolation')
        plt.plot(xold[indxs], yold[indxs], '.')
        indxs = np.argsort(xnew)
        plt.plot(xnew[indxs], ynew[indxs], linewidth=4)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return ynew


def remove_rows(row_indices=[], *args):
    output = ()

    for arg in args:
        if len(np.shape(arg)) == 1:
            output += (np.delete(arg, row_indices),)
        elif len(np.shape(arg)) == 2:
            output += (np.delete(arg, row_indices, axis=0),)
        else:
            print 'object not supported'

    if len(args) == 1:
        return output[0]
    else:
        return output


def hessian_from_Xy(X, y):
    X = np.array(X)
    y = np.array(y)
    ymean = np.mean(y)
    nrows, ncols = X.shape
    sigma = np.std(X, axis=0)
    sigma = np.mean(sigma)
    dy = (y - ymean).reshape((nrows, 1))
    hess = np.dot(X.T, X*dy)/float(nrows)/sigma**4
    return hess


def bwr_2d_plot(matrix, x=[], y=[],
              figsize=(5,3.5),
              vmax=0, vmin=0,
              nlevels=100,
              exponent=1,
              show_colorbar=True,
              figname='bwr 2D plot',
              xlabel='xlabel',
              ylabel='ylabel'):
    import matplotlib.colors as mcolors

    plt.figure(figname, figsize=figsize)

    real_data = np.real(matrix)
    data_max = np.abs(np.max(real_data))
    data_min = np.abs(np.min(real_data))

    # calculate the proportion of positive and negative points:
    nminus = int(data_min/(data_max + data_min)*nlevels)
    nplus = int(data_max / (data_max + data_min)*nlevels)

    # calculate the colors to put white in the middle:
    zero_to_one = np.linspace(0, 1, nminus) ** exponent
    m = np.vstack((zero_to_one, zero_to_one, np.ones(nminus)))
    colors = [tuple(row) for row in m.T]

    one_to_zero = np.linspace(1, 0, nplus) ** exponent
    m = np.vstack((np.ones(nplus), one_to_zero, one_to_zero))
    colors += [tuple(row) for row in m.T]

    cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue',
                                                     colors=colors,
                                                     N=len(colors) - 1,
                                                     )

    if not vmax: vmax = np.max(real_data)
    if not vmin: vmin = np.min(real_data)
    ny, nx = np.shape(real_data)
    if x == []: x = np.arange(nx)
    if y == []: y = np.arange(ny)

    plt.pcolormesh(x, y, real_data, cmap=cmap,
                   shading='gouraud', vmax=vmax, vmin=vmin)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if show_colorbar: plt.colorbar()


def remove_linear_phase(ph, wrap=False):
    phcl = np.copy(ph)
    n = np.size(ph)
    x = np.linspace(-n / 2., n / 2., n)
    y = np.unwrap(phcl)
    m, b = np.polyfit(x, y, 1)
    phcl -= b + m * x
    if wrap: phcl = np.remainder(phcl, 2 * np.pi)
    return phcl


def remove_linear_phase_from_rows(X, wrap=False):
    Xcl = []

    for ph in np.copy(X):
        Xcl.append(remove_linear_phase(ph, wrap))

    return np.array(Xcl)
