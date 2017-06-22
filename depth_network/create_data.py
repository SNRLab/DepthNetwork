import h5py
from h5py import h5d, h5f, h5s, h5p, h5t
import depth_network.common as common
import os.path
import glob


def u(string):
    return bytes(string, 'UTF-8')


def create_virtual_data(file_path, source_files, dataset_name):
    files = list(map(lambda f: h5f.open(u(f), flags=h5f.ACC_RDONLY), source_files))
    print(files)
    datasets = []
    dataspaces = []
    num_elems = 0

    for f in files:
        dataset = h5d.open(f, u(dataset_name))
        datasets.append(dataset)
        dataspace = dataset.get_space()
        dataspace.select_all()
        dataspaces.append(dataspace)
        num_elems += dataspace.shape[0]
        dims = dataspace.shape[1:]

    print(datasets)
    print(dataspaces)
    print(num_elems)

    virtual_file = h5f.create(u(file_path))
    virtual_dataspace = h5s.create_simple((num_elems,) + dims)

    dcpl = h5p.create(h5p.DATASET_CREATE)
    starting_elem = 0
    for file_name, dataset, dataspace in zip(source_files, datasets, dataspaces):
        start = (starting_elem, 0, 0, 0)
        virtual_dataspace.select_hyperslab(start, (1, 1, 1, 1), block=dataspace.shape)
        print(virtual_dataspace.get_select_bounds())
        dcpl.set_virtual(virtual_dataspace, u(file_name), u(dataset_name), dataspace)
        starting_elem += dataspace.shape[0]

    h5d.create(virtual_file, u(dataset_name), h5t.NATIVE_FLOAT, virtual_dataspace, dcpl=dcpl).close()

    virtual_dataspace.close()
    virtual_file.close()

    for file, dataset, dataspace in zip(files, datasets, dataspaces):
        dataspace.close()
        dataset.close()
        file.close()

    f = h5py.File(os.path.join(common.data_dir, 'rgb.hdf5'))

def main():
    rgb_file_names = glob.glob(os.path.join(common.data_dir, "*_rgb_small*.hdf5"))
    depth_file_names = glob.glob(os.path.join(common.data_dir, "*_z_small*.hdf5"))
    create_virtual_data(os.path.join(common.data_dir, 'rgb.hdf5'), rgb_file_names, 'RGB')
    create_virtual_data(os.path.join(common.data_dir, 'depth.hdf5'), depth_file_names, 'Z')

if __name__ == '__main__':
    main()
