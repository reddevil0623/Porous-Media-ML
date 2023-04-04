def PersistenceSum(Bcodes):
    """
    :param Bcodes: persistent barcodes of input images
    :return: feature vectors using sum of barcodes
    """
    import numpy as np
    results=[]

    for i in range(len(Bcodes)):
        sum=0
        for j in range(len(Bcodes[i])):
            sum=sum+ Bcodes[i][j][1]-Bcodes[i][j][0]
        results.append(sum)
    results=np.array(results)
    return results

def AlgebraicFunctions(Bcodes):
    """
    :param Bcodes: persistent barcodes of input images
    :return: feature vectors using algebraic functions
    """
    import numpy as np
    rows = len(Bcodes)
    results=[0]*rows
    for i in range(len(Bcodes)):
        sum1=0
        sum2=0
        sum3=0
        sum4=0
        q=[]
        for k in range(len(Bcodes[i])):
            q.append(Bcodes[i][k][1])
        qmax=max(q)
        for j in range(len(Bcodes[i])):
            sum1 = sum1 + ((Bcodes[i][j][1]-Bcodes[i][j][0])**4)*(Bcodes[i][j][0]**2)
            sum2 = sum2 + (Bcodes[i][j][1]-Bcodes[i][j][0])*Bcodes[i][j][0]
            sum3 = sum3 + (Bcodes[i][j][1]-Bcodes[i][j][0])*(qmax-Bcodes[i][j][1])
            sum4 = sum4 + ((Bcodes[i][j][1]-Bcodes[i][j][0])**4)*((qmax-Bcodes[i][j][1])**2)
        results[i]=[sum1, sum2, sum3, sum4]
    results=np.array(results)
    return results

def Landscape(Pdiagram):
    """
    :param Pdiagram: persistent barcodes of input images
    :return: feature vectors using persistence landscapes
    """
    import numpy as np
    from gtda.diagrams import PersistenceLandscape
    persistence_landscape = PersistenceLandscape(n_layers=1, n_bins=10, n_jobs=-1)
    results = persistence_landscape.fit_transform(Pdiagram)
    results = results.reshape(results.shape[0],results.shape[1]*results.shape[2])
    return results

def SaddleSum(Bcodes0,Bcodes1):
    """
    :param Bcodes0: persistent barcodes of dim 0
    :param Bcodes1: persistent barcodes of dim 1
    :return:
    """
    import numpy as np
    result0 = []
    result1 = []
    for i in range(len(Bcodes0)):
        sum0 = 0
        sum1 = 0

        for j in range(len(Bcodes0[i])):
            sum0 = sum0+ (Bcodes0[i][j][1]-Bcodes0[i][j][0])*Bcodes0[i][j][1]
        result0.append(sum0)
        for k in range(len(Bcodes1[i])):
            sum1 = sum1 + (Bcodes1[i][j][1] - Bcodes1[i][j][0]) * Bcodes1[i][j][0]
        result1.append(sum1)
    result0 = np.array(result0)
    result1 = np.array(result1)
    results = result1-result0
    results = np.array(results).reshape(len(Bcodes0),1)
    return results

def quadrant_separation(Bcodes):
    """

    :param signed_homology: persistent barcodes of input images using signed distance filtration
    :return: separated algebraic vectorization results, use can choose the wanted algebraic functions accordingly
    """
    import numpy as np
    rows = len(Bcodes)
    results = [0] * rows
    for i in range(len(Bcodes)):
        first_sum = 0
        first_sum1 = 0
        first_sum2 = 0
        first_sum3 = 0
        first_sum4 = 0
        third_sum = 0
        third_sum1 = 0
        third_sum2 = 0
        third_sum3 = 0
        third_sum4 = 0
        fourth_sum = 0
        fourth_sum1 = 0
        fourth_sum2 = 0
        fourth_sum3 = 0
        fourth_sum4 = 0
        q1 = [0]
        q4 = [0]
        q3 = [-100]
        for k in range(len(Bcodes[i])):
            if Bcodes[i][k][1]>=0 and Bcodes[i][k][0]>=0:
                q1.append(Bcodes[i][k][1])
            elif Bcodes[i][k][1]>=0 and Bcodes[i][k][0]<=0:
                q4.append(Bcodes[i][k][1])
            elif Bcodes[i][k][1]<=0 and Bcodes[i][k][0]<=0:
                q3.append(Bcodes[i][k][1])
        q1max = max(q1)
        q4max = max(q4)
        q3max = max(q3)
        for j in range(len(Bcodes[i])):
            if Bcodes[i][j][1]>=0 and Bcodes[i][j][0]>=0:
                first_sum = first_sum + Bcodes[i][j][1] - Bcodes[i][j][0]
                first_sum1 = first_sum1 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * (Bcodes[i][j][0] ** 2)
                first_sum2 = first_sum2 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * Bcodes[i][j][0]
                first_sum3 = first_sum3 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * (q1max - Bcodes[i][j][1])
                first_sum4 = first_sum4 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * ((q1max - Bcodes[i][j][1]) ** 2)
            elif Bcodes[i][k][1] >= 0 and Bcodes[i][k][0] <= 0:
                third_sum = third_sum + Bcodes[i][j][1] - Bcodes[i][j][0]
                third_sum1 = third_sum1 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * (Bcodes[i][j][0] ** 2)
                third_sum2 = third_sum2 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * Bcodes[i][j][0]
                third_sum3 = third_sum3 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * (q4max - Bcodes[i][j][1])
                third_sum4 = third_sum4 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * ((q4max - Bcodes[i][j][1]) ** 2)
            elif Bcodes[i][k][1] <= 0 and Bcodes[i][k][0] <= 0:
                fourth_sum = fourth_sum + Bcodes[i][j][1] - Bcodes[i][j][0]
                fourth_sum1 = fourth_sum1 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * (Bcodes[i][j][0] ** 2)
                fourth_sum2 = fourth_sum2 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * Bcodes[i][j][0]
                fourth_sum3 = fourth_sum3 + (Bcodes[i][j][1] - Bcodes[i][j][0]) * (q3max - Bcodes[i][j][1])
                fourth_sum4 = fourth_sum4 + ((Bcodes[i][j][1] - Bcodes[i][j][0]) ** 4) * ((q3max - Bcodes[i][j][1]) ** 2)
        results[i] = [first_sum,third_sum,fourth_sum]#[first_sum,third_sum,fourth_sum] # [ first_sum1, first_sum2, first_sum3, first_sum4, third_sum1, third_sum2, third_sum3, third_sum4, fourth_sum1, fourth_sum2, fourth_sum3, fourth_sum4]#[first_sum, first_sum1, first_sum2, first_sum3, first_sum4, third_sum, third_sum1, third_sum2, third_sum3, third_sum4, fourth_sum, fourth_sum1, fourth_sum2, fourth_sum3, fourth_sum4]
    results = np.array(results)
    return results

def signed_distance_function(u,d,period):
    """
    :param u: images to be calculated
    :param d: grid spacing
    :param period: if identifying sides of the image or not
    :return: the signed distance value for each pixel
    """
    import skfmm
    import numpy as np
    import statistics
    for i in range(u.shape[0]):
        u[i] = u[i]-0.5
        u[i] = -skfmm.distance(u[i] >= 0, dx=d, periodic=period) + skfmm.distance(u[i] < 0, dx=d, periodic=period)
    return u

def ColumnNormalize(Feature):
    """
    :param array of feature vectors
    :return: normalize the vectors by each coordinates
    """
    a = Feature.shape[0]
    b = Feature.shape[1]
    normalized = [[0]*b]*a
    for i in range(b):
        vector = []
        for j in range(a):
            vector.append(Feature[j][i])
        vector = np.array(vector)
        mu=np.mean(vector)
        sd=np.std(vector)
        vector=(vector-mu)/sd
        for k in range(a):
            Feature[k][i] = vector[k]
    return Feature

def GetMean(Feature):
    """
    :param Feature: Feature vectors(from training set)
    :return: (array of means of each coordinate)
    """
    import numpy as np
    a = Feature.shape[0]
    b = Feature.shape[1]
    Means = []
    for i in range(b):
        vector = []
        for j in range(a):
            vector.append(Feature[j][i])
        vector = np.array(vector)
        mu = np.mean(vector)
        Means.append(mu)
    Means = np.array(Means)
    return Means

def GetSd(Feature):
    """

    :param Feature: array of feature vectors
    :return: standard deviation of each column
    """
    import numpy as np
    a = Feature.shape[0]
    b = Feature.shape[1]
    Sd = []
    for i in range(b):
        vector = []
        for j in range(a):
            vector.append(Feature[j][i])
        vector = np.array(vector)
        mu = np.std(vector)
        Sd.append(mu)
    Sd = np.array(Sd)
    return Sd

def ColumnNormalize(Feature, Means, Sd):
    """
    :param array of feature vectors
    :return: normalize the vectors by each coordinates
    """
    a = Feature.shape[0]
    b = Feature.shape[1]
    normalized = [[0]*b]*a
    for i in range(b):
        for j in range(a):

            if Sd[i] == 0:
                Sd[i] = 1 # when standard deviation is zero, (x-mu)/1 would be able to remove the effect of that column
            Feature[j][i] = (Feature[j][i]-Means[i])/Sd[i]
    return Feature
