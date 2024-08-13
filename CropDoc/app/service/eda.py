from datetime import datetime

import matplotlib.pyplot as plt
from loguru import logger

from app.config import AppConfig
from app.utility.path import join_path


def plot_class_frequency_histogram(y,
                                   x,
                                   x_label: str = 'Frequency Count',
                                   y_label: str = "Classes",
                                   title: str = "Frequency Counts for Classes",
                                   save_path: str = '/output/'):
    """ Plot a histogram of the frequency counts for classes in a dataset.

    Args:
        y (list): List of classes.
        x (): List of frequency counts for each class.
        x_label (str, optional): The plot label for x axis. Defaults to 'Frequency Count'.
        y_label (str, optional): The plot label for y axis. Defaults to "Classes".
        title (str, optional): The title of the plot. Defaults to "Frequency Counts for Classes".
        save_path (str, optional): The absolute save path. Defaults to '/output/'.

    Raises:
        ValueError: If the save path does not end in a .png file.
        e: Any other exception that occurs during the plot creation.
    """
    try:

        # If the save path is the default, create a new file name
        if save_path == '/output/':
            now_string = datetime.now().strftime("%Y%m%d-%H%M%S")
            file_name = f'class_frequency_histogram_{now_string}.png'
            save_path = join_path(AppConfig.DATA_DIR, 'output', 'eda',
                                  file_name)
        else:

            # Check that the save path ends in a .png file
            if not save_path.endswith('.png'):
                logger.error("Save path must end in a .png file")
                raise ValueError("Save path must end in a .png file")

        # Create the plot
        plt.figure(figsize=(10, 5))
        bars = plt.bar(y, x)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(rotation=45)

        # Add the frequency count to the top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height,
                     f'{int(height)}',
                     ha='center',
                     va='bottom')

        plt.tight_layout()

        # Save the plot to output directory
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        logger.error(f"Error creating plot: {e}")
        raise e
