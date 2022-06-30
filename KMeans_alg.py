import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import operator
import sys
import time

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

def separator():
    print('\n\n\n')
    print('#'*120)
    print('#'*120)
    print('\n\n\n')


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
"""
K-means is a very simple clustering algorithm that tries to partition the input data in
k clusters. K-means works by iteratively refining an initial estimate of class centroids
as follows:
1. Initialize centroids µi, i = 1 ...k, randomly or with some guess.
2. Assign each data point to the class ci of its nearest centroid.
3. Update the centroids as the average of all data points assigned to that class.
4. Repeat 2 & 3 until convergence.
K-means tries to minimize the total within-class variance
"""
def input_way():

    try:
        choice = int(input("please enter 0 for a points saved in excel file, or 1 for manually inputting:\nDefault 1 \n"))
    except ValueError or KeyboardInterrupt:
        choice = 1

    if choice == 0:
        path = input("please enter file path:\n")
        data = pd.read_excel(path)
        z = int(input("please enter number of columns you want to get: "))
        cols = []
        n = 0   # count_pts
        s = 0

        print('\n')
        for i in range(0,z):

            col = input("column names: ")
            s += 1
            cols.append(col)


        df = pd.DataFrame(data, columns=cols)

        for index, row in df.iterrows():
            n += 1

        print("\n\ndata points: \n {}".format(df))


    elif choice == 1:
        all_pts = []
        separated_pts = []

        n = int(input("please enter number of points to use: "))
        s = int(input("please enter shape of each point: "))
        for i in range(1,n+1):
            cols = []
            for j in range(1, s+1):
                p = float(input("p{}=".format(i)))
                all_pts.append(p)
                if len(all_pts) % s == 0:
                    separated_pts.append(all_pts)
                    all_pts = []
            print('\n')

        for i in range(0,s):
            col = input("column names: ")
            cols.append(col)

        df = pd.DataFrame(separated_pts, columns=[cols])
        print("\n\ndata points: \n {}".format(df))

        try:
            save = int(input("please enter 1 for saving your df in an excel file, or 0 for not saving:\nDefault 0 \n"))
        except ValueError or KeyboardInterrupt:
            save = 0

        if save == 0:
            print("\nYour inputted datapoints won't be saved..\n\n")
        elif save == 1:
            name = input("\nPlease Enter file name:\n")
            df.to_excel("created data_points/"+name+".xlsx")

    return df, n, s


df, n_points, point_features = input_way()
#df = list(df)
#print(df)
# E:\3rd_year\KOLLIA\CI tasks\points.xlsx
# random.choice()
# how to split a list by a uniform strip using python
# print(df.shape)
# sample = df.sample()
separator()

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################




def rows(df, point_features):

    cols_names = []
    data_point = []

    for column in df:
        cols_names.append(column)

    for index, row in df.iterrows():
        data_point.append((row[0:point_features].tolist()))      # row is a pandas series that converted to a list

    return cols_names, data_point


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

cols_names, data_point = rows(df, point_features)
# point = [point for point in data_point]
# print(type(point))    # list

print("\npoints: \n", data_point)
print("\ncol_num: \n", len(cols_names))
print("\nNames_of_cols: ", cols_names)

separator()
time.sleep(1)

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


def sampling_centroids(data_point):
    #num_clusters=2

    try:
        num_clusters = int(input("Please enter number of clusters: "))

    except ValueError or KeyboardInterrupt:
        num_clusters = 2


    if num_clusters > n_points:
        print("n_clusters nust be < your data points number!\n")
        centroids, num_clusters = sampling_centroids(data_point)

    elif  num_clusters<1:
        print("n_clusters nust be >= 1!\n")
        centroids, num_clusters = sampling_centroids(data_point)
        #sys.exit()


    centroids = []
    c = 0
    i = 0
    while i in range(0, num_clusters+c):
        print("\n\nNum_Iterations: ", num_clusters+c)
        print("Loop Incrementerd by: ", c)
        print("Remaining_iter: ", num_clusters+c-i, end='\n\n')
        print('------------------------------------------------------------')

        sample = random.choice(data_point)
        print("\ncentroid_sample: ", sample, end='\n')

        if not sample in centroids:
            centroids.append(sample)

        elif sample in centroids:
            c += 1
            print("\nrepeated_cluster_centroids: ", c, end='\n')
        i+=1
    #print("centroids: ", centroids, '\n')
    return centroids, num_clusters


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################


centroids, num_clusters = sampling_centroids(data_point)
print("\nCentroids: ", centroids)
print("\nnum_clusters: ", len(centroids))
print("\nType_clusters: ", type(centroids))

separator()
time.sleep(1)



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################




def eacleadian_distance(data_point, centroids):
    row_diff = []
    all_rows_diff = []

    for c in centroids:
        for row in data_point:
            diff = tuple(map(operator.sub, row, c))   # diff. bet. each centroid and all data_points
            #print("\n", diff)
            row_diff.append(diff)
            if len(row_diff) % len(data_point) == 0:
                all_rows_diff.append(row_diff)
                row_diff = []
    #print("\nall_rows_diff = ", all_rows_diff)

    diff_type = [d_t for d in all_rows_diff for d_t in d]
    # list   , list

    # print(type(diff_type))    # list
    # print(diff_type)

    #print(sample, '\n')
    #print(all_rows_diff)
    #return all_rows_diff
    #print("\nall_rows_diff: ", all_rows_diff)
    #print("\nlen_all_rows_diff: ", len(all_rows_diff), end='\n')





    count = 0
    c = 0
    all = []
    points = []
    points_lst = []
    all_points = []

    # powering each element by 2
    pow_all = [x**2 for row in all_rows_diff for d in row for x in d]

    for x in pow_all:
        all.append(x)
        count += 1
        if count % point_features == 0:
            points.append(all)
            all = []

    for p in points:
        c += 1
        # print("P type: ", type(p), end='\n')
        points_lst.append(p)
        if c % n_points == 0:
            all_points.append(points_lst)
            points_lst = []




    # taking sqrt for each data_point to get distance from centroids
    sum = 0
    count = 0
    diff_sum = []
    eacleadian_dist = []


    num = [n for lst in all_points for l in lst for n in l]
    #print("\n\nnum: ", num, end='\n')

    for i in num:
        count += 1
        sum += i
        if count % point_features == 0:
            diff_sum.append(math.sqrt(sum))
            sum = 0
            if count % n_points == 0:
                eacleadian_dist.append(diff_sum)
                diff_sum = []

    return eacleadian_dist

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



eacleadian_dist = eacleadian_distance(data_point, centroids)
print("\n\neacleadian_dist: ", eacleadian_dist, end='\n')
print("\nlen_eacleadian_dist: ", len(eacleadian_dist), end='\n')

separator()
time.sleep(1)


############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



def clustering(eacleadian_dist, centroids):

    centers1 = {}
    cols = []
    cluster_point = {}
    counter = {}
    centroids_1 = {}
    centroids_2 = {}
    index_lst = []

    for i, c in enumerate(centroids):
        centroids_1[i] = c
    print("\nCentroids_1: ", centroids_1)


    centroids_tuple = [tuple(centroid) for centroid in centroids]
    print("\nCentroids_tuple:", centroids_tuple)


    for i, x in enumerate(centroids):
        #print(i, ": ", x)
        centers1[i] = eacleadian_dist[i]


        table = pd.DataFrame.from_dict(centers1)
        cols = table.columns.tolist()        # cluster indecies

        nearest_pnt = table.min(axis=1)
        #group = table.idxmin(axis=1)
        group = table.idxmin(axis=1).tolist()  # points to be clusterd

    for p in group:
        if p in cols:
            count = group.count(p)
            counter[p] = count
            #count = 0

            if len(counter) % len(cols) == 0:
                break

    counter_tuple = counter.items()
    #print("\n\ncounter_tuple:\n\n ", counter_tuple)


    counter_lst_of_tuples = []
    for k, v in counter_tuple:
        entry = (k, v)
        counter_lst_of_tuples.append(entry)
        #print("\nEntry_type: ", type(entry))     #tuple
    #print("\n\ncounter_lst_of_tuples:\n\n ", counter_lst_of_tuples)


    # sorted to make zip by index with the real centroids
    sorted_counter_lst_of_tuples = []
    for i in range(0, len(counter_lst_of_tuples)):
        for t in counter_lst_of_tuples:
            if i == t[0]:
                sorted_counter_lst_of_tuples.append(t)
    #print("\n\nsorted_counter_lst_of_tuples:\n\n ", sorted_counter_lst_of_tuples)


    #arranged_counter_tuple = counter_tuple.sort(key=lambda tup:tup[0])
    #print("\n\narranged_counter_tuple:\n\n ", arranged_counter_tuple)

    counts_lst = []
    for t in sorted_counter_lst_of_tuples:
        counts_lst.append(t[-1])

    centers_n_pts = dict(zip(centroids_tuple, counts_lst))


    return centers1, table, nearest_pnt, cols, group, counter, sorted_counter_lst_of_tuples, centers_n_pts


"""
    num_of_pts = []
    centroid_pt = []
    for key in centroids_1.keys():
        for k in counter.keys():
            if key == k:
                num_of_pts.append(counter.values())
                centroid_pt.append(centroids_1.values())
                centers_n_pts = dict(zip(num_of_pts, centroid_pt))
    # the number of points for each centroid


        for i in ordered_counter_keys:
            for key in counter.keys():
                if i == key:
                    ordered_counts.append(counter[key])
        counter = dict(zip(ordered_counter_keys, ordered_counts))

"""

############################################################################################################
############################################################################################################


centers1, table, nearest_pnt, cols, group, counter, sorted_counter_lst_of_tuples, centers_n_pts = clustering(eacleadian_dist, centroids)
print("\nCenters1: ", centers1)
print("\n\nData:\n\n ", table)
print("\n\nnearest_pnt:\n\n", nearest_pnt)
print("\n\ngroup:\n\n ", cols)
print("\n\ngroup for each element in order:\n\n ", group)
print("\n\ncounter:\n\n ", counter)
print("\n\nsorted_counter_lst_of_tuples:\n\n ", sorted_counter_lst_of_tuples)
print("\n\nNum_points closest to each centroid:\n\n ", centers_n_pts, end='\n\n')

separator()
time.sleep(1)



############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
# E:\3rd_year\KOLLIA\CI tasks\points.xlsx
"""
    for i, pt in enumerate(data_point):
        idxs.append(i)
        pts.append(pt)

    data_pt = dict(zip(idxs, pts))    # points and indicies


    for i in cols:
        for j in group:
            if i == j:
                points_clusters[i] =


        for i in cols:
            for j in data_pt.values():
                if i == j:
                    pts.append(data_pt.keys())
        print("group: ", pts)


"""
def mean_of_points(group, data_point):
    idxs = []

    for i, pt in enumerate(data_point):
        idxs.append(tuple(pt))

    data_pt = dict(zip(idxs, group))
    #pts_table = pd.DataFrame.from_dict(data_pt)

    data_pt_lst = data_pt.items()
    x = [i for i in data_pt_lst]


    # new_centroids = np.mean(points, axis=0)
    return data_pt, data_pt_lst


data_pt, data_pt_lst = mean_of_points(group, data_point)
print("\npoints indicies and clusters: \n", data_pt, end='\n\n')
print("\ndata_pt_lst:\n\n", data_pt_lst, end='\n\n')
#print(len(data_pt_lst))    # dict_items
#x = [r for t in data_pt_lst for r in t ]
#print('\n', x, end='\n')    # tuple

#print("\npoints indicies and clusters table: \n", pts_table)

#print("\n\nnew_centroids: ", tuple(new_centroids))
#print("\n\nlen_new_centroids: ", len(new_centroids))

separator()
time.sleep(1)




############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



def new_Centroids(data_pt_lst, cols):
    elements = []

    for x in data_pt_lst:
        elements.append(x)
        #print(x[-1])
        #print(type(x))      # tuple

    #print(elements)
    # print(type(elements))   # list

    each_centroid = []
    cluster = []
    for col in cols:
        for x in elements:
            if x[-1] == col:
                #print(x[0])
                each_centroid.append((x[0]))

        cluster.append(each_centroid)
        each_centroid = []


    arrs = []
    new_centroids = []
    for cls in cluster:
        for lst in cls:
            arrs.append(np.array(lst))

        new_centroids.append(sum(arrs) / len(arrs))
        arrs = []


    new_centroids_lst = []
    for centroid in new_centroids:
        centroid = centroid.tolist()
        new_centroids_lst.append(centroid)


    return cluster, new_centroids, new_centroids_lst


############################################################################################################
############################################################################################################


cluster, new_centroids, new_centroids_lst = new_Centroids(data_pt_lst, cols)
print("\nEach cluster as a list of tuples:\n ", cluster, end='\n\n')
print("\nNum_clusters: ", len(cluster), end='\n\n')

print("\nNew Centroids:\n", new_centroids, end='\n\n\n')
#print("\nType_New_Centroids:\n", type(new_centroids), end='\n\n\n')

print("\nnew_centroids_lst: ", new_centroids_lst, end='\n')
#print("new_centroids_lst type: ", type(new_centroids_lst), end='\n\n')


separator()
time.sleep(1)





############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
# E:\3rd_year\KOLLIA\CI tasks\KMeans\points.xlsx

def update_cluster_centroids(data_point, new_centroids_lst):
    group_of_groups = []

    try:
        iter = int(input("\nPlease enter number of iteration for KMeans algorithm (default=50): "))
    except ValueError or KeyboardInterrupt:
        iter = 50

    for i in range(1, iter+1):
        last_distance = eacleadian_distance(data_point, new_centroids_lst)
        print("last_distance:\n", last_distance, end='\n\n')
        separator()



        centers1, table, nearest_pnt, cols, group, counter, sorted_counter_lst_of_tuples, centers_n_pts = clustering(last_distance, new_centroids_lst)
        print("\nCenters1: ", centers1)
        print("\n\nData:\n\n ", table)
        print("\n\nnearest_pnt:\n\n", nearest_pnt)
        print("\n\ngroup:\n\n ", cols)
        print("\n\ngroup for each element in order:\n\n ", group)
        print("\n\ncounter:\n\n ", counter)
        print("\n\nsorted_counter_lst_of_tuples:\n\n ", sorted_counter_lst_of_tuples)
        print("\n\nNum_points closest to each centroid:\n\n ", centers_n_pts, end='\n\n')
        separator()
        time.sleep(.5)

        group_of_groups.append(group)

        #if group not in group_of_groups:
        #    group_of_groups.append(group)
        #elif group in group_of_groups:
        #if len(group_of_groups)>5 and group_of_groups[-1] == group_of_groups[-2] == group_of_groups[-3] == group_of_groups[-4] == group_of_groups[-5]:
        if len(group_of_groups)>2 and group_of_groups[-1] == group_of_groups[-2]:
            print(f"\nAll data clustering till convergence: {group_of_groups}")
            print(f"\nAlgorithm stops at {i} iteration.\n")
            break

        data_pt, data_pt_lst = mean_of_points(group, data_point)
        print("\npoints indicies and clusters: \n", data_pt, end='\n\n')
        separator()


        cluster, new_centroids, new_centroids_lst = new_Centroids(data_pt_lst, cols)
        print("\nEach cluster as a list of tuples: ", cluster, end='\n\n')
        print("\nNum_clusters: ", len(cluster), end='\n\n')
        separator()


        print("\nNew Centroids:\n", new_centroids, end='\n\n\n')
        print("\nType_New_Centroids:\n", type(new_centroids), end='\n\n\n')
        print("\nnew_centroids_lst: ", new_centroids_lst, end='\n')
        print("new_centroids_lst type: ", type(new_centroids_lst), end='\n\n')

    return new_centroids_lst, group
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

new_centroids_lst, labels = update_cluster_centroids(data_point, new_centroids_lst)
new_centroids_lst = pd.DataFrame(new_centroids_lst)
print(type(new_centroids_lst))
labels = np.array(group)
print(f"labels: {labels}")
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################



def plot(df, centroid, labels):
    try:
        try:
            plotting = int(input("\nPlease enter 1 to plot, or zero to not plotting (default=1): "))
        except ValueError or KeyboardInterrupt:
            plotting = 1

        if plotting==1:
            f, axes = plt.subplots(1, 2, figsize=(11, 5))
            axes[0].scatter(df.iloc[:,0], df.iloc[:,1], alpha=0.5)
            axes[1].scatter(df.iloc[:,0], df.iloc[:,1], c=labels, alpha=0.5)
            axes[1].scatter(centroid[0], centroid[1], marker="*", s=100, c='g')
            #axes[1].scatter(centroid[0], centroid[1], marker=".", s=100, c='r')

            plt.show()

        elif plotting==0:
            print('\n\nEnd of code without plotting----------------', end='\n')
    except:

        print('\n\nError: plot', end='\n')


############################################################################################################
############################################################################################################

plot(df, new_centroids_lst, labels)
