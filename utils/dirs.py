import os

def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    if isinstance(dirs, str):
        try:
            dir_ = dirs
            if not os.path.exists(dir_):
                    os.makedirs(dir_)
            return 0
        except Exception as err:
            print("Creating directories error: {0}".format(err))
            exit(-1)
    else:
        try:
            for dir_ in dirs:
                if not os.path.exists(dir_):
                    os.makedirs(dir_)
            return 0
        except Exception as err:
            print("Creating directories error: {0}".format(err))
            exit(-1)