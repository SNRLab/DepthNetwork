import h5py
from h5py import h5d, h5f, h5s, h5p, h5t
import depth_network.common as common
import os.path
import glob


def u(string):
    return bytes(string, 'UTF-8')


def create_virtual_data(file_path, source_files, dataset_name):
    files = list(map(lambda f: h5f.open(u(f), flags=h5f.ACC_RDONLY), source_files))
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


def main():
    rgb_file_names = list(map(lambda n: os.path.join(common.data_dir, n),
                              ['0_rgb_small.hdf5', '2_rgb_small.hdf5', '3_rgb_small.hdf5', '4_rgb_small.hdf5',
                               '5_rgb_small.hdf5', '6_rgb_small.hdf5', '7_rgb_small.hdf5', '8_rgb_small.hdf5',
                               '9_rgb_small.hdf5', '10_rgb_small.hdf5', '11_rgb_small.hdf5', '12_rgb_small.hdf5',
                               '13_rgb_small.hdf5', '14_rgb_small.hdf5', '15_rgb_small.hdf5', '16_rgb_small.hdf5']))
    depth_file_names = list(map(lambda n: os.path.join(common.data_dir, n),
                                ['0_z_small.hdf5', '2_z_small.hdf5', '3_z_small.hdf5', '4_z_small.hdf5',
                                 '5_z_small.hdf5', '6_z_small.hdf5', '7_z_small.hdf5', '8_z_small.hdf5',
                                 '9_z_small.hdf5', '10_z_small.hdf5', '11_z_small.hdf5', '12_z_small.hdf5',
                                 '13_z_small.hdf5', '14_z_small.hdf5', '15_z_small.hdf5', '16_z_small.hdf5']))
    create_virtual_data(os.path.join(common.data_dir, 'rgb.hdf5'), rgb_file_names, 'RGB')
    create_virtual_data(os.path.join(common.data_dir, 'depth.hdf5'), depth_file_names, 'Z')


if __name__ == '__main__':
    main()
