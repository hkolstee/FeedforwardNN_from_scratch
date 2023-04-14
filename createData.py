import numpy as np
import random
import matplotlib.pyplot as plt

def createDataCircular(nr_class1, nr_class2):
    outer_radius = 1
    inner_radius = 0.5
    
    # angle, radius
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class1,))
    radius = np.random.uniform(low = inner_radius * 1.1, high = outer_radius, size = (nr_class1,))
    
    # create class 1 (outer ring)
    class1_x = radius * np.cos(alpha)
    class1_y = radius * np.sin(alpha)
    
    # new random angles and radius for class 2
    alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = (nr_class2,))
    radius = np.random.uniform(low = 0, high = inner_radius * 0.9, size = (nr_class2,))
    
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

def createDataCircular2(nr_class1, nr_class2):
    # init classes
    class1_x = []
    class1_y = []
    class2_x = []
    class2_y = []
    
    for i in np.arange(0, 1, 0.2):
        outer_radius = i + 0.02
        inner_radius = i + 0.01

        if (i % 0.4 == 0):
            # angle, radius
            alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = round(nr_class1 * i,))
            radius = np.random.uniform(low = inner_radius, high = outer_radius, size = (round(nr_class1 * i),))
            
            # create class 1 
            class1_x = np.concatenate((class1_x, radius * np.cos(alpha)), axis = 0)
            class1_y = np.concatenate((class1_y, radius * np.sin(alpha)), axis = 0)
        else:
            # angle, radius
            alpha = np.random.uniform(low = 0, high = 2 * np.pi, size = round(nr_class2 * i,))
            radius = np.random.uniform(low = inner_radius, high = outer_radius, size = (round(nr_class2 * i),))
            
            # create class 2
            class2_x = np.concatenate((class2_x, radius * np.cos(alpha)), axis = 0)
            class2_y = np.concatenate((class2_y, radius * np.sin(alpha)), axis = 0)

    # reshape
    class1 = np.column_stack((class1_x, class1_y, np.zeros((len(class1_x), ))))
    class2 = np.column_stack((class2_x, class2_y, np.ones((len(class2_x), ))))

    # concat data
    data = np.vstack((class1, class2))

    # shuffle data
    np.random.shuffle(data)

    return data

data = createDataCircular2(150, 150)