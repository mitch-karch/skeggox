from sklearn.model_selection import train_test_split
import pandas as pd


def test_spliting_result(save=False):
    alldf = pd.read_csv(f'train_or.csv')

    y = alldf['label']
    trndf = pd.DataFrame()
    tstdf = pd.DataFrame()
    valdf = pd.DataFrame()
    # this is temp holder for now to hold both tr and val
    tr_val_df = pd.DataFrame()

    # Split the training dataset into a training and a validation

    # NOTE: pick some training for testing first!! (TODO we need to be able to load new data
    # so better to have a separate test from train/val to avoid re-run for whole set)
    tr_val_df['filename'], tstdf['filename'], tr_val_df['label'], tstdf['label'] = \
        train_test_split(alldf['filename'], y, test_size=0.15,
                         random_state=4, stratify=y)
    y = tr_val_df['label']
    trndf['filename'], valdf['filename'], trndf['label'], valdf['label'] = \
        train_test_split(tr_val_df['filename'], y, test_size=0.15,
                         random_state=4, stratify=y)

    print('alldf:', alldf.shape, 'train:', trndf.shape,
          'val:', valdf.shape, 'test', tstdf.shape)
    labels = [0, 1, 3, 5, 10, 11, 12]
    dfs = {'train': trndf, 'val': valdf, 'test': tstdf}
    for df_key in dfs:
        df = dfs.get(df_key)
        total = df.shape[0]
        print(df_key, 'total:', total)
        [print(df_key, df[df['label'] == l].shape[0], l, df[df['label'] == l].shape[0]/total)
         for l in labels]

    if save:
        tr_val_df.to_csv('train.csv')
        tstdf.to_csv('test.csv')


# RUN to see a random split result
test_spliting_result(save=True)
