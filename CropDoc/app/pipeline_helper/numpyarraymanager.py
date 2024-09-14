import numpy as np
from loguru import logger

class NumpyArrayManager:
    
    @staticmethod
    def check_arrays_are_exact(a: np.array, b: np.array) -> bool:
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError(f"Both inputs must be numpy arrays, got {type(a)} and {type(b)}")
        return np.array_equal(a, b)
    
    @staticmethod
    def append_missing_unique_elements(existing_arr: np.array, new_arr: np.array) -> tuple:
        """
        Take the existing array and new array, work through the new array and for any elements
        that don't exist in the existing array, append them to the end of existing array

        Args:
            existing_arr (np.array): An array of existing elements, which are all unique
            new_arr (np.array): The new array to be merged with existing array

        Returns:
            np.array: The updated array with new elements appended to the end
            np.array: The indexes of the appended elements from new array
        """
        logger.debug(f"Appending missing unique elements from new array to existing array")
        if not isinstance(existing_arr, np.ndarray) or not isinstance(new_arr, np.ndarray):
            raise TypeError(f"Both inputs must be numpy arrays, got {type(existing_arr)} and {type(new_arr)}")

        # Create a set of existing elements for faster lookup
        existing_set = set(existing_arr.flat)
        
        # Find elements in new_arr that are not in existing_set
        mask = np.array([x not in existing_set for x in new_arr.flat])
        unique_new_elements = new_arr[mask]
        
        if unique_new_elements.size > 0:
            updated_arr = np.concatenate((existing_arr, unique_new_elements))
            appended_indexes = np.where(mask)[0]
            return updated_arr, appended_indexes
     
        return existing_arr, np.array([], dtype=int)
    
    @staticmethod
    def get_difference(a: np.array, b: np.array) -> np.array:
        """Find the set difference of two arrays.

        Return the unique values in b that are not in a.
        Args:
            a (np.array): The first array
            b (np.array): The second array

        Returns:
            np.array: The difference between the two arrays
        """
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError(f"Both inputs must be numpy arrays, got {type(a)} and {type(b)}")
        return np.setdiff1d(b, a)