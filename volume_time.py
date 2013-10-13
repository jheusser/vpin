from itertools import izip_longest
import itertools
import pandas
import numpy as np

all_trades = pandas.read_csv('./april_trades.csv', parse_dates=[0], index_col=0)
usd_trades = all_trades[all_trades['d.currency'] == 'USD']

volume = (usd_trades['d.amount_int'])
trades = (usd_trades['d.price_int'])

def cleanup(x):
    if isinstance(x, str) and 'e-' in x:
        return 0
    else:
        return float(x)

volume = volume.apply(lambda x: cleanup(x))
volume = volume.astype(float32)

##### 
typestr = (usd_trades['d.type_str'])
typestr[typestr == 'bid'] = 0 
typestr[typestr == 'ask'] = 1

trades_1min = trades.resample('1min').diff(1).dropna()
volume_1min = volume.resample('1min', how='sum')

# assign trade sign to 1 minute time bar by averaging buys and sells and taking the more common one
typestr_1min = typestr.astype(float32).resample('1min',how='mean').round()

df = pandas.DataFrame({'type': typestr_1min, 'volume': volume_1min})
df_trades = pandas.DataFrame({'volume': volume_1min, 'trades': trades_1min})

# volume time!
delta_p_expanded = []
missed = 0
for t in df.itertuples():
    idx = t[0]
    side = t[1]
    vol = t[2]

    if np.nan_to_num(vol) == 0.0:
        continue

    # expand price change over standardised volume
    for i in range(0, int(vol)):
        # 1 unit trades 
        delta_p_expanded.append((idx, side))

# side for each standard size trade
expanded = pandas.DataFrame.from_records(delta_p_expanded, index=0)

#####################
# return distribution for volume time sampling
# volume time!
volume_sample_trades_expanded = []
missed = 0
for t in df_trades.itertuples():
    idx = t[0]
    vol = t[2]
    delta_p = t[1]

    if np.nan_to_num(vol) == 0.0:
        continue

    # expand price change over standardised volume
    for i in range(0, int(vol)):
        # 1 unit trades 
        volume_sample_trades_expanded.append((idx, delta_p))

trades_expanded = pandas.DataFrame.from_records(volume_sample_trades_expanded, index=0)

################################


def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


# volume in BTC which makes up one bucket
n_bucket_size = 500.0

# find single-period VPIN
OI = []
start = 0 
for each in grouper(n_bucket_size, expanded[1]):
  slce = pandas.Series(each)
  counts = slce.value_counts()
  if len(counts) > 1:
      OI.append(np.abs(counts[1] - counts[0])/n_bucket_size)
  else:
      if 0 in counts:
          OI.append(counts[0]/n_bucket_size)
      else:
          OI.append(counts[1]/n_bucket_size)



# find time boundaries for volume buckets
buckets = []
V = n_bucket_size 
running_volume = 0.0
start_idx = None 

for idx in expanded.index:
    if not start_idx:
        start_idx = idx 

    if running_volume >= V:
        buckets.append((start_idx, idx))

        start_idx = None
        running_volume = 0
    running_volume += 1

# find mid time of volume buckets
mid_buckets = []
for start,end in buckets:
  diff = end - start
  mid_buckets.append(start + (diff/2))

# volume bucket duration
diffs = []
for start,end in buckets:
  diffs.append(end-start)
 
vpin_df = pandas.rolling_mean(pandas.Series(OI[:-1], index=mid_buckets), window=500)
trades_adj = trades.resample('1min').reindex_like(vpin_df, method='ffill')

#######
## Plot VPIN vs Trades
import matplotlib as mpl
mpl.rc('font', **{'sans-serif':'Verdana','family':'sans-serif','size':8})
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['axes.linewidth'] = 0.75


fig, axes = plt.subplots(nrows=2, ncols=1)
plt.subplots_adjust(hspace = 0.5)

vpin_df.plot(ax=axes[0])  
axes[0].set_title('VPIN')

trades_adj.plot(ax=axes[1])
axes[1].set_title('Trades')

fig.tight_layout()


#####
## Get mid price series from ticker for the same period

all_ticker = pandas.read_csv('./all_ticker.txt', parse_dates=[0], index_col=0)

ticker_df = all_ticker.ix[vpin_df.index[0] : vpin_df.index[-1]]
# calculate mid
ticker_df = ticker_df.resample('1min').apply(axis=1, func=lambda s: (s['d.bid'] + s['d.ask'])/2)

# align with VPIN
ticker_df = ticker_df.reindex_like(vpin_df, method='ffill')

### plot of return distributions of sampling by volume time (more normal).
plt.figure()
# volume-time samples price returns
p1 = trades_expanded[1].hist(normed=True, bins=45, alpha=0.3)
# trade-time sampled price returns
p2 =ticker_df.diff(1).hist(normed=True, bins=45, alpha=0.3)
p2.legend(['Volume Time', 'Chronological'])

plt.draw()


### plot overlay of VPIN and trades

ax = pandas.DataFrame({'VPIN': vpin_df , 'Price': trades_adj.fillna(method='ffill')}).plot(secondary_y=['VPIN'])
ax.set_title('Price vs VPIN')
ax.right_ax.set_ylabel('Probability of Informed Trading')
plt.draw()

