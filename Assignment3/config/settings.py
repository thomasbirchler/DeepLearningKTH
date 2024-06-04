FILENAMES_TRAIN = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]
FILENAMES_TEST = ['test_batch']

GDparams = {
    #'lambda': 2.15714286e-03,
    'lambda': 5e-03,
    'n_batch': 5,
    'eta_min': 0.00001,
    'eta_max': 0.1,
    'n_epochs': 65,
    #'cycles': 3,
    #'eta_s': 800,
    'update_steps': 23000,
    'batch_size': 64,
    'eta_s': 5 * 49000 / 64
}
