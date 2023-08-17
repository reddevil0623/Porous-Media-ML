def AddColumn(M, a):
    # Check if input is a 3D numpy array
    if len(M.shape) != 3:
        raise ValueError("Input matrix M is not 3D")

    # Get the dimensions of the 2D matrices in M
    _, rows, _ = M.shape

    # Create a 2D array of shape (rows, 1) with all elements equal to a
    column = np.full((rows, 1), a)

    # Add column to both ends of each 2D matrix in M
    M_new = np.array([np.hstack((column, m, column)) for m in M])

    return M_new

def Connection(M):
    # Check the shape of M
    if len(M.shape) != 3 or M.shape[2] != 3:
        raise ValueError("Input array M must have shape (x, y, 3)")

    # Initialize a list to store the results for each persistence diagram
    results = []

    # Iterate through the persistence diagrams
    for diagram in M:
        # Extract birth and death values
        birth_values = diagram[:, 0]
        death_values = diagram[:, 1]

        # Find indices of barcodes with birth value -10000
        indices = np.where(birth_values == -10000)[0]

        # Extract the death values of the barcodes with birth value -10000
        death_values_of_interest = death_values[indices]

        # Find the minimum death value that is not -10000
        result = np.min(death_values_of_interest[death_values_of_interest != -10000])

        # Append the result to the list of results
        results.append(result)

    return results

def PersistenceSum(barcode_list):
    """
    Calculates the persistence sum for a given list of barcodes.

    The persistence sum is obtained by iterating through each barcode in the list, 
    and for each barcode, subtracting the start value from the end value and summing 
    the results. This function returns a NumPy array containing the sum for each barcode set.

    :param barcode_list: List of barcodes, where each barcode is represented as a tuple (start, end).
    :return: NumPy array containing the sum for each barcode set.
    """
    import numpy as np
    
    results = []

    # Iterate through the barcode sets
    for barcode_set in barcode_list:
        total_sum = 0
        
        # Iterate through the individual barcodes and calculate the sum of (end - start)
        for barcode in barcode_set:
            total_sum += barcode[1] - barcode[0]

        results.append(total_sum)

    return np.array(results)


def AlgebraicFunctions(barcode_list):
    """
    Calculates feature vectors using algebraic functions on the given persistent barcodes.

    The feature vectors are obtained by computing four sums for each barcode set:
      - sum1: Sum of ((end - start)**4) * (start**2) for each barcode
      - sum2: Sum of (end - start) * start for each barcode
      - sum3: Sum of (end - start) * (qmax - end) for each barcode
      - sum4: Sum of ((end - start)**4) * ((qmax - end)**2) for each barcode
    where qmax is the maximum end value in the barcode set.

    :param barcode_list: List of barcodes, where each barcode is represented as a tuple (start, end).
    :return: NumPy array containing the feature vectors [sum1, sum2, sum3, sum4] for each barcode set.
    """
    import numpy as np
    
    rows = len(barcode_list)
    results = [0] * rows

    # Iterate through the barcode sets
    for i, barcode_set in enumerate(barcode_list):
        sum1, sum2, sum3, sum4 = 0, 0, 0, 0
        q = [barcode[1] for barcode in barcode_set]
        qmax = max(q)

        # Iterate through the individual barcodes and calculate the sums
        for barcode in barcode_set:
            start, end = barcode
            difference = end - start
            sum1 += (difference ** 4) * (start ** 2)
            sum2 += difference * start
            sum3 += difference * (qmax - end)
            sum4 += (difference ** 4) * ((qmax - end) ** 2)

        results[i] = [sum1, sum2, sum3, sum4]

    return np.array(results)


def Landscape(Pdiagram):
    """
    Calculates feature vectors using persistence landscapes for the given persistent barcodes.

    The persistence landscapes are calculated using the gtda.diagrams.PersistenceLandscape 
    class. This function reshapes the resulting landscapes into a 2D array where each row 
    represents a feature vector for a corresponding barcode set.

    :param Pdiagram: 3D array representing persistent barcodes of input images, where each 
                     barcode is represented as a tuple (start, end).
    :return: 2D NumPy array containing the feature vectors for each barcode set, using 
             persistence landscapes.
    """
    import numpy as np
    from gtda.diagrams import PersistenceLandscape

    # Create a PersistenceLandscape object with specific parameters
    persistence_landscape = PersistenceLandscape(n_layers=1, n_bins=10, n_jobs=-1)

    # Fit and transform the persistence diagrams to calculate persistence landscapes
    results = persistence_landscape.fit_transform(Pdiagram)

    # Reshape the results into a 2D array, where each row represents a feature vector
    results = results.reshape(results.shape[0], results.shape[1] * results.shape[2])

    return results


def SaddleSum(Bcodes0, Bcodes1):
    """
    Calculates the difference between two sums obtained from two sets of persistent barcodes (dim 0 and dim 1).

    For each set of barcodes, the function computes the sum of (end - start) * end for dim 0 barcodes 
    and the sum of (end - start) * start for dim 1 barcodes. It then returns the difference between 
    the sums for dim 1 and dim 0 as a reshaped NumPy array.

    :param Bcodes0: List of persistent barcodes of dimension 0, where each barcode is represented as a tuple (start, end).
    :param Bcodes1: List of persistent barcodes of dimension 1, similar to Bcodes0.
    :return: NumPy array containing the differences between the sums for dimensions 1 and 0, reshaped to (len(Bcodes0), 1).
    """
    import numpy as np
    result0 = []
    result1 = []

    # Iterate through the barcode sets for dimension 0
    for i in range(len(Bcodes0)):
        sum0 = 0
        sum1 = 0

        for j in range(len(Bcodes0[i])):
            sum0 += (Bcodes0[i][j][1] - Bcodes0[i][j][0]) * Bcodes0[i][j][1]
        result0.append(sum0)

        # Iterate through the barcode sets for dimension 1 (fixed the loop variable from j to k)
        for k in range(len(Bcodes1[i])):
            sum1 += (Bcodes1[i][k][1] - Bcodes1[i][k][0]) * Bcodes1[i][k][0]  # Fixed the index from j to k
        result1.append(sum1)

    result0 = np.array(result0)
    result1 = np.array(result1)
    results = result1 - result0
    results = results.reshape(len(Bcodes0), 1)

    return results


def quadrant_separation(Bcodes):
    """
    Separates persistent barcodes into three quadrants and calculates various sums.

    The function identifies barcodes belonging to three quadrants:
    - Quadrant 1 (Q1): Both start and end values are greater than or equal to 0.
    - Quadrant 4 (Q4): Start values are less than or equal to 0, and end values are greater than or equal to 0.
    - Quadrant 3 (Q3): Both start and end values are less than or equal to 0.

    Several sums are calculated for each quadrant, and the results are returned as a 2D NumPy array.

    :param Bcodes: List of persistent barcodes of input images using signed distance filtration.
    :return: NumPy array containing separated algebraic vectorization results.
    """
    import numpy as np
    rows = len(Bcodes)
    results = [0] * rows

    for i in range(len(Bcodes)):
        # Initialize sums for each quadrant
        first_sum, third_sum, fourth_sum = 0, 0, 0
        # Other sums can be initialized similarly if needed
        q1, q4, q3 = [0], [0], [-100]

        # Separate barcodes into quadrants
        for k in range(len(Bcodes[i])):
            if Bcodes[i][k][1] >= 0 and Bcodes[i][k][0] >= 0:
                q1.append(Bcodes[i][k][1])
            elif Bcodes[i][k][1] >= 0 and Bcodes[i][k][0] <= 0:
                q4.append(Bcodes[i][k][1])
            elif Bcodes[i][k][1] <= 0 and Bcodes[i][k][0] <= 0:
                q3.append(Bcodes[i][k][1])

        q1max, q4max, q3max = max(q1), max(q4), max(q3)

        # Calculate sums for each quadrant
        for j in range(len(Bcodes[i])):
            # Logic for Quadrant 1
            if Bcodes[i][j][1] >= 0 and Bcodes[i][j][0] >= 0:
                first_sum += Bcodes[i][j][1] - Bcodes[i][j][0]
                # Additional calculations for Quadrant 1 can be added here

            # Logic for Quadrant 4
            elif Bcodes[i][j][1] >= 0 and Bcodes[i][j][0] <= 0:
                third_sum += Bcodes[i][j][1] - Bcodes[i][j][0]
                # Additional calculations for Quadrant 4 can be added here

            # Logic for Quadrant 3
            elif Bcodes[i][j][1] <= 0 and Bcodes[i][j][0] <= 0:
                fourth_sum += Bcodes[i][j][1] - Bcodes[i][j][0]
                # Additional calculations for Quadrant 3 can be added here

        # Choose the required algebraic functions by uncommenting the desired line
        results[i] = [first_sum, third_sum, fourth_sum]

    results = np.array(results)
    return results


def signed_distance_function(u, d, period):
    """
    Calculates the signed distance value for each pixel in a given set of images.

    The function takes a 3D array representing multiple images, grid spacing, and a periodic flag. 
    It applies the signed distance transform to each image using the Scikit-FMM library, and 
    returns the modified images with the calculated signed distance values.

    :param u: 3D array representing images to be calculated (shape: number of images x height x width).
    :param d: Grid spacing used in the distance transform.
    :param period: Boolean value indicating if identifying sides of the image or not (periodic boundaries).
    :return: 3D array with the signed distance value for each pixel in the input images.
    """
    import skfmm
    import numpy as np

    # Iterate through the images in the input array
    for i in range(u.shape[0]):
        # Subtract 0.5 from all pixel values
        u[i] -= 0.5

        # Compute positive and negative distance transforms and combine them to obtain signed distance
        positive_distance = skfmm.distance(u[i] >= 0, dx=d, periodic=period)
        negative_distance = skfmm.distance(u[i] < 0, dx=d, periodic=period)
        u[i] = -positive_distance + negative_distance

    return u

def GetMean(Feature):
    """
    Calculates the mean of each coordinate (column) in the feature vectors.

    :param Feature: 2D array of feature vectors.
    :return: 1D array containing the means of each coordinate.
    """
    import numpy as np
    return np.mean(Feature, axis=0)


def GetSd(Feature):
    """
    Calculates the standard deviation of each coordinate (column) in the feature vectors.

    :param Feature: 2D array of feature vectors.
    :return: 1D array containing the standard deviations of each coordinate.
    """
    import numpy as np
    return np.std(Feature, axis=0)


def ColumnNormalize(Feature, Means, Sd):
    """
    Normalizes the feature vectors by each coordinate using precomputed mean and standard deviation values.

    :param Feature: 2D array of feature vectors.
    :param Means: 1D array containing the means of each coordinate.
    :param Sd: 1D array containing the standard deviations of each coordinate.
    :return: 2D array with normalized feature vectors.
    """
    import numpy as np
    a, b = Feature.shape
    for i in range(b):
        if Sd[i] == 0:
            Sd[i] = 1 # Prevent division by zero
        Feature[:, i] = (Feature[:, i] - Means[i]) / Sd[i]
    return Feature

