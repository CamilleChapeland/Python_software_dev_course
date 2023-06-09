"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.

Functions include:
        load_csv - Load a numpy array from a csv
        daily_mean - Calculate the daily mean of a 2D inflammation data array.
        daily_max - Calculate the daily max of a 2D inflammation data array.
        daily_min - Calculate the daily min of a 2D inflammation data array.
"""
import numpy as np

def load_csv(filename):
    """Load a Numpy array from a CSV
    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array for each day.

       :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
       :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily maximum of a 2D inflammation data array for each day.

      :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
      :returns: An array of max values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily minimum of a 2D inflammation data array for each day.

      :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
      :returns: An array of minimum values of measurements for each day.
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """Normalise patient data from a 2D inflammation data array."""
    max = np.max(data, axis=0)
    return data / max[:, np.newaxis]


def daily_std(data, axis=0):
    """Calculate the daily standard deviation of a 2D inflammation data array for each day.

          :param data: A 2D data array with inflammation data (each row contains measurements for a single patient across all days).
          :returns: An array of standard deviation values of measurements for each day.
    """
    return np.std(data, axis=0)
