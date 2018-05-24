import os
import pandas as pd
import traceback


def create_file(file_name ="name", data=""):
    dir_name = os.path.dirname(file_name)
    if not does_directory_exist(dir_name):
        os.makedirs(dir_name)
    fp = open(file_name, "w+")
    fp.write(data)
    fp.close()


def does_directory_exist(directory_name='folder/'):
    """
    relative folder starts at the current working directory (i.e., the
    directory where the application resides).
    """
    import os.path
    folder = os.getcwd()
#    folder = os.path.join(folder, relative_folder)
    dir_path = os.path.join(folder, directory_name)
    return os.path.isdir(dir_path)


def delete_file(file_name="name"):
    try:
        if os.path.exists(file_name): os.remove(file_name)
    except:
        traceback.print_exc()


# def delete_file(file_name="name"):
#     try:
#         os.remove(file_name)
#     except:
#         pass


def create_directory_if_needed(fname="scraps/asins/test.csv"):
    dir_name = os.path.dirname(fname)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def initialize_csv(fname='', column_labels=[]):
    create_directory_if_needed(fname)
    df = pd.DataFrame(columns=column_labels)
    df.to_csv(fname, index=False)


def append_row_to_csv(row=[], fname=''):
    srow = pd.DataFrame([row])
    with open(fname, 'a') as f:
        srow.to_csv(f, header=False, index=False)
