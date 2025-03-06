import h5py

file_path = "/scratch2/mrenaudin/colorlessgreenRNNs/datah5/wikitext103_ccgtagged.train.hdf5"

with h5py.File(file_path, "r") as f:
    print("Keys in the HDF5 file:")
    for key in f.keys():
        print(key)
