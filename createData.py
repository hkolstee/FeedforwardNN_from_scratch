import numpy as np
import random
import matplotlib.pyplot as plt

def createDataCircular(nr_class1, nr_class2):
    outer_radius = 1
    inner_radius = 0.5
    
    # angle, radius
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class1,))
    radius = np.random.uniform(low = inner_radius * 1.05, high = outer_radius, size = (nr_class1,))
    
    # create class 1 (outer ring)
    class1_x = radius * np.cos(alpha)
    class1_y = radius * np.sin(alpha)
    
    # new random angles and radius for class 2
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class2,))
    radius = np.random.uniform(low = 0, high = inner_radius * 0.95, size = (nr_class2,))
    
    # create class 2 (inner circle)
    class2_x = radius * np.cos(alpha)
    class2_y = radius * np.sin(alpha)

    # reshape
    class1 = np.column_stack((class1_x, class1_y, np.zeros((nr_class1, ))))
    class2 = np.column_stack((class2_x, class2_y, np.ones((nr_class1, ))))

    # concat data
    data = np.vstack((class1, class2))

    # shuffle data
    np.random.shuffle(data)

    return data


def createDataLinear(nr_class1, nr_class2):    
    # create class 1 
    class1_x = np.random.uniform(low = -1, high = 1, size=(nr_class1,))
    class1_y = np.random.uniform(low = -1, high = -0.05, size=(nr_class1,))

    # create class 2  
    class2_x = np.random.uniform(low = -1, high = 1, size=(nr_class1,))
    class2_y = np.random.uniform(low = 0.05, high = 1, size=(nr_class1,))

    # reshape
    class1 = np.column_stack((class1_x, class1_y, np.zeros((nr_class1, ))))
    class2 = np.column_stack((class2_x, class2_y, np.ones((nr_class1, ))))

    # concat data
    data = np.vstack((class1, class2))

    # shuffle data
    np.random.shuffle(data)

    return data