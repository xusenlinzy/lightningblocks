import shutil
import time

from .logger import logger, tqdm


def get_path_from_url(url, root_dir, check_exist=True, decompress=True):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        decompress (bool): decompress zip or tar file. Default is `True`
    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    import os.path
    import os
    import tarfile
    import zipfile

    def is_url(path):
        """
        Whether path is URL.
        Args:
            path (string): URL string or not.
        """
        return path.startswith('http://') or path.startswith('https://')

    def _map_path(url, root_dir):
        # parse path after download under root_dir
        fname = os.path.split(url)[-1]
        fpath = fname
        return os.path.join(root_dir, fpath)

    def _get_download(url, fullname):
        import requests
        # using requests.get method
        fname = os.path.basename(fullname)
        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            logger.info("Downloading {} from {} failed with exception {}".format(
                fname, url, str(e)))
            return False

        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code "
                               "{}!".format(url, req.status_code))

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = fullname + "_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024, unit='KB') as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)

        return fullname

    def _download(url, path):
        """
        Download from url, save to path.
        url (str): download url
        path (str): download to given path
        """

        if not os.path.exists(path):
            os.makedirs(path)

        fname = os.path.split(url)[-1]
        fullname = os.path.join(path, fname)
        retry_cnt = 0

        logger.info("Downloading {} from {}".format(fname, url))
        DOWNLOAD_RETRY_LIMIT = 3
        while not os.path.exists(fullname):
            if retry_cnt < DOWNLOAD_RETRY_LIMIT:
                retry_cnt += 1
            else:
                raise RuntimeError("Download from {} failed. "
                                   "Retry limit reached".format(url))

            if not _get_download(url, fullname):
                time.sleep(1)
                continue

        return fullname

    def _uncompress_file_zip(filepath):
        with zipfile.ZipFile(filepath, 'r') as files:
            file_list = files.namelist()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)

            elif _is_a_single_dir(file_list):
                # `strip(os.sep)` to remove `os.sep` in the tail of path
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)

                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)
                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _is_a_single_file(file_list):
        if len(file_list) == 1 and file_list[0].find(os.sep) < 0:
            return True
        return False

    def _is_a_single_dir(file_list):
        new_file_list = []
        for file_path in file_list:
            if '/' in file_path:
                file_path = file_path.replace('/', os.sep)
            elif '\\' in file_path:
                file_path = file_path.replace('\\', os.sep)
            new_file_list.append(file_path)

        file_name = new_file_list[0].split(os.sep)[0]
        for i in range(1, len(new_file_list)):
            if file_name != new_file_list[i].split(os.sep)[0]:
                return False
        return True

    def _uncompress_file_tar(filepath, mode="r:*"):
        with tarfile.open(filepath, mode) as files:
            file_list = files.getnames()

            file_dir = os.path.dirname(filepath)

            if _is_a_single_file(file_list):
                rootpath = file_list[0]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            elif _is_a_single_dir(file_list):
                rootpath = os.path.splitext(file_list[0].strip(os.sep))[0].split(
                    os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                files.extractall(file_dir)
            else:
                rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
                uncompressed_path = os.path.join(file_dir, rootpath)
                if not os.path.exists(uncompressed_path):
                    os.makedirs(uncompressed_path)

                files.extractall(os.path.join(file_dir, rootpath))

            return uncompressed_path

    def _decompress(fname):
        """
        Decompress for zip and tar file
        """
        logger.info("Decompressing {}...".format(fname))

        # For protecting decompressing interupted,
        # decompress to fpath_tmp directory firstly, if decompress
        # successed, move decompress files to fpath and delete
        # fpath_tmp and remove download compress file.

        if tarfile.is_tarfile(fname):
            uncompressed_path = _uncompress_file_tar(fname)
        elif zipfile.is_zipfile(fname):
            uncompressed_path = _uncompress_file_zip(fname)
        else:
            raise TypeError("Unsupport compress file type {}".format(fname))

        return uncompressed_path

    assert is_url(url), "downloading from {} not a url".format(url)
    fullpath = _map_path(url, root_dir)
    if os.path.exists(fullpath) and check_exist:
        logger.info("Found {}".format(fullpath))
    else:
        fullpath = _download(url, root_dir)

    if decompress and (tarfile.is_tarfile(fullpath) or
                       zipfile.is_zipfile(fullpath)):
        fullpath = _decompress(fullpath)

    return fullpath
