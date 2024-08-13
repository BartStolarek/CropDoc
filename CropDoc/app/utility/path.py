import os

from app.config import AppConfig


def resolve_path(path):
    """Resolve and return the absolute path."""
    return os.path.abspath(os.path.expanduser(path))


def is_file(path):
    """Check if a path points to a file."""
    return os.path.isfile(resolve_path(path))


def is_directory(path):
    """Check if a path points to a directory."""
    return os.path.isdir(resolve_path(path))


def file_exists(path):
    """Check if a file exists at the given path."""
    return os.path.exists(resolve_path(path)) and os.path.isfile(
        resolve_path(path))


def directory_exists(path):
    """Check if a directory exists at the given path."""
    return os.path.exists(resolve_path(path)) and os.path.isdir(
        resolve_path(path))


def create_dir_if_not_exists(directory):
    """Create a directory if it does not already exist."""
    resolved_dir = resolve_path(directory)
    if not os.path.exists(resolved_dir):
        os.makedirs(resolved_dir)


def list_files(directory, extension=None):
    """
    List files in a directory optionally filtered by extension.
    
    Args:
        directory (str): Directory path to list files from.
        extension (str, optional): Filter files by extension (e.g., '.txt'). Defaults to None.
    
    Returns:
        list: List of file names.
    """
    resolved_dir = resolve_path(directory)
    file_list = []
    for file in os.listdir(resolved_dir):
        if extension:
            if file.endswith(extension):
                file_list.append(file)
        else:
            file_list.append(file)
    return file_list


def join_path(*paths):
    """
    Join multiple paths ensuring no duplication of common parts.
    
    Args:
        *paths (str): Variable number of paths to join.
    
    Returns:
        str: Joined path.
    """
    return os.path.join(*paths)


def combine_path(*paths):
    """
    Combine multiple paths ensuring no duplication of common parts.
    
    Args:
        *paths (str): Variable number of paths to combine.
    
    Returns:
        str: Combined path.
    """
    # Resolve all paths to their absolute forms
    resolved_paths = [resolve_path(path) for path in paths]

    # Initialize the combined path with the first resolved path
    combined_path = resolved_paths[0]

    # Iterate through remaining paths to combine
    for path in resolved_paths[1:]:
        # Check if the current combined path is already a subdirectory of the new path
        if path.startswith(combined_path):
            combined_path = path
        elif combined_path.startswith(path):
            continue  # Skip if current combined path is already a subdirectory
        else:
            # Otherwise, combine the paths
            combined_path = os.path.join(combined_path, path.lstrip(os.sep))

    return combined_path


def absolute_path(path):
    """Return the absolute path of a given path."""
    return os.path.abspath(resolve_path(path))


def parent_directory(path):
    """Return the parent directory of a given path."""
    return os.path.dirname(resolve_path(path))


def file_name(path):
    """Return the base name of a file path."""
    return os.path.basename(resolve_path(path))


def directory_is_empty(path):
    """Check if a directory is empty."""
    return not os.listdir(resolve_path(path))


def split_extension(file_name):
    """Split the file name and extension."""
    return os.path.splitext(file_name)


def get_root_directory():
    """Provide the path to root directory of the project"""
    return AppConfig.ROOT_DIR


def get_cropdoc_directory():
    """Provide the path to the CropDoc directory"""
    return AppConfig.CROPDOC_DIR


def get_app_directory():
    """Provide the path to the app directory"""
    return AppConfig.APP_DIR


def find_directory(path, name):
    """Find a directory with a given name in the path."""
    for root, dirs, files in os.walk(resolve_path(path)):
        if name in dirs:
            return os.path.join(root, name)
    return None
