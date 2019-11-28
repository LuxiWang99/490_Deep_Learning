import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
import pickle 
import os
import librosa
import pandas as pd
import sys
from pathlib import Path
from pydub import AudioSegment

warnings.resetwarnings()

SR = 44100 # Sample Rate
TRAIN_SENTENCE_LENGTH = 3 # Sentence Length in Seconds
BATCH_SIZE = 15

'''
arguments:
    path_to_fma_wav_files : str
        path the directory containing the fma mp3 files
    path_to_data : str
        path to the directory to store .npy and .pkl files in
    
output of main (all in the data directory):
    - .npy files for senteces
    - song_IDs.pkl containing list of IDs
    - labels.pkl containing dict of IDs to period ("Ren, "Bar", "Cla", "Rom", "Mod")
    - hyperparameters.pkl containing dict of SR, SENTENCE_LENGTH, BATCH_SIZE, WORD_SIZE
'''
def main():
    # command line arguments
    path_to_fma_mp3_files = sys.argv[1:][0]
    path_to_musicnet_wav_files = sys.argv[1:][1]
    path_to_data = sys.argv[1:][2]
    print('Using FMA Path: ' + path_to_fma_mp3_files)
    print('Using MusicNet Path: ' + path_to_musicnet_wav_files)
    print('Using Data Path: ' + path_to_data)

    path_to_data_train = Path(path_to_data, 'train')
    path_to_data_test = Path(path_to_data, 'test')

    musicnet_ids = [1727,1728,1729,1730,1733,1734,1735,1739,1742,1749,1750,1751,1752,1755,1756,1757,1758,1759,1760,1763,1764,1765,1766,1768,1771,1772,1773,1775,1776,1777,1788,1789,1790,1791,1792,1793,1805,1807,1811,1812,1813,1817,1818,1819,1822,1824,1828,1829,1835,1859,1872,1873,1876,1893,1916,1918,1919,1922,1923,1931,1932,1933,2075,2076,2077,2078,2079,2080,2081,2082,2083,2104,2105,2106,2112,2113,2114,2116,2117,2118,2119,2127,2131,2138,2140,2147,2148,2149,2150,2151,2154,2155,2156,2157,2158,2159,2160,2161,2166,2167,2168,2169,2177,2178,2179,2180,2186,2191,2194,2195,2196,2198,2200,2201,2202,2203,2204,2207,2208,2209,2210,2211,2212,2213,2214,2215,2217,2218,2219,2220,2221,2222,2224,2225,2227,2228,2229,2230,2231,2232,2234,2237,2238,2239,2240,2241,2242,2243,2244,2247,2248,2282,2283,2284,2285,2288,2289,2292,2293,2294,2295,2296,2297,2298,2300,2302,2303,2304,2305,2307,2308,2310,2313,2314,2315,2318,2319,2320,2322,2325,2330,2334,2335,2336,2341,2342,2343,2345,2346,2348,2350,2357,2358,2359,2364,2365,2366,2368,2371,2372,2373,2374,2376,2377,2379,2381,2382,2383,2384,2388,2389,2390,2391,2392,2393,2397,2398,2403,2404,2405,2406,2410,2411,2415,2416,2417,2420,2422,2423,2424,2431,2432,2433,2436,2441,2442,2443,2444,2451,2462,2463,2466,2471,2472,2473,2476,2477,2478,2480,2481,2482,2483,2486,2487,2488,2490,2491,2492,2494,2497,2501,2502,2504,2505,2506,2507,2509,2510,2512,2514,2516,2521,2522,2523,2527,2528,2529,2530,2531,2532,2533,2537,2538,2540,2542,2550,2555,2556,2557,2560,2562,2564,2566,2567,2568,2570,2571,2572,2573,2575,2576,2581,2582,2586,2588,2590,2591,2593,2594,2595,2596,2603,2607,2608,2611,2614,2618,2619,2620,2621,2622,2626,2627,2628,2629,2632,2633,2659,2677,2678]

    musicnet_test_ids = np.random.choice(musicnet_ids, 75, replace=False)

    next_id = 1
    train_song_IDs = []
    test_song_IDs = []
    labels = {}

    # First Process FMA
    # filename = "track_info.pkl"
    # with open(filename, 'rb') as handle:
    #     fma_info = pickle.load(handle)

    # fma_info = fma_info[pd.notnull(fma_info['track', 'period'])]

    # for index, row in fma_info.iterrows():
    #     filename = str(index)
    #     period = row['track', 'period']
    #     if (period != "Ren"):
    #         print("Processing: " + filename)
    #         while(len(filename) < 6):
    #             filename = str(0) + filename  
    #         file_dir = filename[:-3]

    #         src = Path(path_to_fma_mp3_files, file_dir, filename + '.mp3')
    #         dst = str(src)[:-4] + '.wav'

    #         sound = AudioSegment.from_mp3(src)
    #         wav_file = open(dst,'wb')
    #         sound.export(wav_file, format="wav")
    #         wav_file.close()

    #         print("   Made .wav file.")

    #         x, sr = librosa.load(dst, sr=SR)
    #         sentences, label = train_wav2sentences(x, period)
    #         for i in range(len(sentences)):
    #             data_file = str(next_id) + '.npy'
    #             data_file_dst = path_to_data_train / data_file
    #             np.save(data_file_dst, sentences[i])
    #             train_song_IDs.append(next_id)
    #             labels[next_id] = label[i]
    #             next_id += 1
    #             print("   Made " + data_file)
    #     else:
    #         print("Will not process rennaissance music: " + filename)

    # Now process music_net_data 
    composer_to_period = {
            "Schubert": "Rom", 
            "Mozart": "Cla", 
            "Dvorak": "Rom",
            "Cambini": "Cla", 
            "Haydn": "Cla", 
            "Brahms": "Rom", 
            "Faure": "Rom", 
            "Ravel": "Mod", 
            "Bach": "Bar", 
            "Beethoven": "Cla"
        }

    metadata_url = "https://homes.cs.washington.edu/~thickstn/media/musicnet_metadata.csv"
    metadata = pd.read_csv(metadata_url)

    metadata['period'] = metadata ['composer'].map(composer_to_period)

    id_to_period = metadata.set_index('id')['period'].to_dict()

    for root, dirs, files in os.walk(path_to_musicnet_wav_files):
        for file in files:
            if file.endswith(".wav"):
                x, sr = librosa.load(os.path.join(root, file), sr=SR)
                filename, file_ext = os.path.splitext(file)
                if np.isin(int(filename), musicnet_test_ids):
                    print("Processing Test File: " + str(file))
                    data_file = str(next_id) + '.npy'
                    data_file_dst = path_to_data_test / data_file
                    np.save(data_file_dst, x)
                    test_song_IDs.append(next_id)
                    labels[next_id] = id_to_period[int(filename)]
                    print("   Made " + data_file)
                    next_id += 1
                else:
                    print("Processing: " + str(file))
                    sentences, label = train_wav2sentences(x, id_to_period[int(filename)])
                    for i in range(len(sentences)):
                        data_file = str(next_id) + '.npy'
                        data_file_dst = path_to_data_train / data_file
                        np.save(data_file_dst, sentences[i])
                        train_song_IDs.append(next_id)
                        labels[next_id] = label[i]
                        print("   Made " + data_file)
                        next_id += 1

    # save the song_IDs list pkl and labels dict pkl
    train_song_id_path = path_to_data_train / 'song_ids.pkl'
    train_song_id_file = open(train_song_id_path,'wb')
    pickle.dump(train_song_IDs, train_song_id_file)
    train_song_id_file.close()  
    print("Saved: " + str(train_song_id_path))

    test_song_id_path = path_to_data_test / 'song_ids.pkl'
    test_song_id_file = open(test_song_id_path,'wb')
    pickle.dump(test_song_IDs, test_song_id_file)
    test_song_id_file.close()  
    print("Saved: " + str(test_song_id_path))

    labels_path = Path(path_to_data, 'labels.pkl')
    labels_file = open(labels_path,'wb')
    pickle.dump(labels, labels_file)
    labels_file.close()   
    print(str(labels_path))

    hyperparameters = {
        'SR': SR,
        'TRAIN_SENTENCE_LENGTH': TRAIN_SENTENCE_LENGTH,
        'BATCH_SIZE': BATCH_SIZE,
    }

    hyperparameters_path = Path(path_to_data, 'hyperparameters.pkl')
    hyperparameters_file = open(hyperparameters_path,'wb')
    pickle.dump(hyperparameters, hyperparameters_file)
    hyperparameters_file.close()   
    print(str(hyperparameters_path))
    
# given a raw numpy array, split to SENTENCE_LENGTH second sentence
def train_wav2sentences(nums, label):
    length = TRAIN_SENTENCE_LENGTH * SR
    final = [nums[i * length:(i + 1) * length] for i in range((len(nums) + length - 1) // length )][:-1]
    if len(final) >= 1:
        labels = [label for i in range(len(final))]
        final = np.array(final)
        print(final.shape)
        return final, labels
    return [], []

# Processes all data files in a given path
'''
src : string
    path to the directory containing wav files to make .npy sentence files for
target_path: string
    directory to place the .npy files in

'''
def process_data(src, target_path, start_idx, song_IDs, labels):
    # with open(target_path, 'rb') as handle:
    #     id2label = pickle.load(handle)
    for root, dirs, files in os.walk(src):
        for file in files:
            x, sr = librosa.load(os.path.join(root, file), sr=SR)
            filename, file_ext = os.path.splitext(file)
            X_temp, y_temp = wav2sentences(x, id2label[int(filename)])
            for x in X_temp:
                X.append(x)
            for label in y_temp:
                y.append(label)
    d = {"data": X, "label": y}
    with open("data.pickle", 'wb') as handle:
        pickle.dump(d, handle)

def map_regex(dict, text):

    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())), flags=re.IGNORECASE)

    if regex.search(text):
        ret = regex.search(text)
        return dict[ret.group()]
    else:
        return None

def process_fma_metadeta(fma_metadata_csv_path):
    df = pd.read_csv(fma_metadata_csv_path, header=None)

    # getting names for columns and index:
    cnames = zip(df.iloc[0,1:], df.iloc[1,1:])
    inames = list(df.iloc[0,:1])   

    #drop the rows with column names (for columns and index)
    df.drop([0,1],axis=0,inplace=True)
    #set the indexes
    df.set_index([0],inplace=True)
    # set the names for columns and indexes
    df.columns = pd.MultiIndex.from_tuples(cnames)
    df.index.names = inames

    pd.set_option('display.max_rows', 1000)
    #classical = df[df['track', 'genre_top'] == 'Classical']
    classical = df[df['track', 'is_classical'] == True]
    print(len(classical))

    info = pd.concat([classical['track', 'composer'], classical['track', 'title']], axis=1)
    print("initial data length: ", len(info[info['track', 'composer'].notnull()]))

    contain_composers = [r'.*Bach.*', r'.*Brahms.*', r'.*Mozart.*', r'.*Schubert.*', r'.*Beethoven.*', r'.*Paganini.*', r'.*Chopin.*', r'.*Grieg.*', r'.*Debussy.*', r'.*Mussorgsky.*', r'.*Liszt.*', r'.*Rachmaninoff.*', r'.*Schumann.*', r'.*Mendelssohn.*', r'.*Alkan.*', r'.*Vivaldi.*', r'.*Wagner.*', r'.*Satie.*', r'.*Camille.*', r'.*Pachelbel.*', r'.*Palestrina.*', r'.*Pizzetti.*', r'.*Bizet.*', r'.*Tchaikovsky.*', r'.*Dvorak.*', r'.*Handel.*', r'.*Bartok.*', r'.*Corelli.*', r'.*Albinoni.*', r'.*Ravel.*', r'.*Spohr.*', r'.*Cambini.*', r'.*Haydn.*', r'.*Faure.*', r'.*Schumann.*', r'.*Kirschner.*', r'.*Purcell.*', r'.*Sigismondo.*', r'.*Merula.*', r'.*Strozzi.*', r'.*Frescobaldi.*', r'.*Paganini.*', r'.*Gibbons.*', r'.*Byrd.*', r'.*Papa.*', r'.*Monteverdi.*', r'.*Lobo.*', r'.*Tallis.*', r'.*Rore.*', r'.*Stabile.*', r'.*Ferrabosco.*', r'.*Dvořák.*', r'.*Victoria.*', r'.*Animuccia.*', r'.*Ives.*', r'.*Janáček.*', r'.*Wolf.*', r'.*Danzi.*']
    composers = ['Bach', 'Brahms', 'Mozart', 'Schubert', 'Beethoven', 'Paganini', 'Chopin', 'Grieg', 'Debussy', 'Mussorgsky', 'Liszt', 'Rachmaninoff', 'Schumann', 'Mendelssohn', 'Alkan', 'Vivaldi', 'Wagner', 'Satie', 'Camille', 'Pachelbel', 'Palestrina', 'Pizzetti', 'Bizet', 'Tchaikovsky', 'Dvorak', 'Handel', 'Bartok', 'Corelli', 'Albinoni', 'Ravel', 'Spohr', 'Cambini', 'Haydn', 'Faure', 'Schumann', 'Kirschner', 'Purcell', 'Sigismondo', 'Merula', 'Strozzi', 'Frescobaldi', 'Paganini', 'Gibbons', 'Byrd', 'Papa', 'Monteverdi', 'Lobo', 'Tallis', 'Rore', 'Stabile', 'Ferrabosco', 'Dvořák', 'Victoria', 'Animuccia', 'Ives', 'Janáček', 'Wolf', 'Danzi']
    composer_to_period = {
        "Albinoni": "Bar",
        "Alkan": "Rom",
        "Animuccia": "Ren",
        "Bach": "Bar",
        "Bartok": "Mod",
        "Beethoven": "Cla",
        "Bizet": "Rom",
        "Brahms": "Rom",
        "Byrd": "Bar",
        "Cambini": "Cla",
        "Camille": "Rom", # Saint-Saens?
        "Chopin": "Rom",
        "Corelli": "Bar",
        "Danzi": "Cla",
        "Debussy": "Rom",
        "Dvorak": "Rom",
        "Dvořák": "Rom",
        "Faure": "Rom",
        "Ferrabosco": "Bar",
        "Frescobaldi": "Bar",
        "Gibbons": "Bar",
        "Grieg": "Rom",
        "Handel": "Cla",
        "Haydn": "Cla",
        "Ives": "Mod",
        "Janáček": "Rom",
        "Kirschner": "Mod", #Experimental Classical
        "Liszt": "Rom",
        "Lobo": "Bar",
        "Mendelssohn": "Rom",
        "merula": "Bar",
        "Monteverdi": "Bar",
        "Mozart": "Cla",
        "Mussorgsky": "Rom",
        "Pachelbel": "Cla",
        "Paganini": "Cla",
        "Paganini": "Cla",
        "Palestrina": "Ren",
        "Papa": "Ren",
        "Pizzetti": "Mod",
        "Purcell": "Bar",
        "Rachmaninoff": "Rom",
        "Ravel": "Mod",
        "Rore": "Ren",
        "Satie": "Rom",
        "Schubert": "Rom",
        "Schumann": "Rom",
        "Schumann": "Rom",
        "Sigismondo": "Bar", #Sigismondo D'India
        "Spohr": "Rom",
        "Stabile": "Ren",
        "Strozzi": "Bar",
        "Tallis": "Ren",
        "Tchaikovsky": "Rom",
        "Victoria": "Ren",
        "Vivaldi": "Cla",
        "Wagner": "Rom",
        "Wolf": "Rom"
    }

    info['track', 'comp_from_title'] = info['track', 'title'].replace(contain_composers, composers, regex=True)
    info['track', 'comp_from_title'].loc[~info['track', 'comp_from_title'].isin(composers)] = np.NaN
    info['track', 'composer'] = info['track', 'composer'].fillna(info['track', 'comp_from_title'])
    info['track', 'period'] = info['track', 'composer'].apply(lambda v: map_regex(composer_to_period, str(v)))

    print("final data length: ", len(info[info['track', 'composer'].notnull()]))
    print("period length: ", len(info[info['track', 'period'].notnull()]))

    print(info['track', 'composer'].unique())

    print(info)
    info.to_csv("track_info.csv")

    file_path = 'track_info.pkl'
    labels_file = open(file_path,'wb')
    pickle.dump(info, labels_file)
    labels_file.close()

if __name__ == "__main__":
    main()