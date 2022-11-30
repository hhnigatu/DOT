import sys
import shutil


def _download_local(src, dest):
    shutil.copytree(src, dest)


def _download_gdrive(src, dest):
    if "google.colab" not in sys.modules:
        raise Exception("You can only use this function in a Google Colab notebook.")
    from google.colab import drive

    drive.mount("/content/drive")


def download_dataset(src, dest="./", type="web"):
    """Download a dataset to a folder on this computer. No return value.

    Params:
    - dest: The path to download the dataset to. Dot (`.`) means current directory. Use
        slashes to separate consecutive folder names.
    - type: One of `web`, `sharepoint`, `gdrive`, or `local`"""
    if type == "local":
        return _download_local(src, dest)
    elif type == "gdrive":
        return _download_gdrive(src, dest)
    else:
        raise NotImplementedError
