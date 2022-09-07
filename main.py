# https://www.youtube.com/watch?v=szyGiObZymo
import math
import json
import os
import librosa

DATASET_PATH = "genre_dataset_reduced"
JASON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # measured in second
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # build a dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)   # 1.2 =B 2 round to higher int number
    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_components = dirpath.split("/")  # genre/blues =B ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing {}".format(semantic_label))

            # process files for a specific genre
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extractoing mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s  # s=0 =B 0
                    finish_sample = start_sample + num_samples_per_segment  # s=0 =B num_samples_per_segment

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)
                        print("{}, segment:{}".format(file_path, s))
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

#___________________________________ MAIN ____________________________
if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JASON_PATH, num_segments=10)

