import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as color
import numpy as np
from collections import Counter
import pandas as pd
import math
from sklearn.cluster import KMeans
from PIL import Image
import webcolors
import json
import argparse

##在启动主函数之前，我们将创建一个ArgumentParser（）对象以接受命令行参数，并创建相应的变量以接受命令行参数的值。
# 与此同时保留了两个“可选”命令行参数，即clusters和imagepath。
parser = argparse.ArgumentParser()

parser.add_argument("--clusters", help="No. of clusters")
parser.add_argument("--imagepath", help="Path to input image")

args = parser.parse_args()

IMG_PATH = args.imagepath if args.imagepath else "D:/fengshuilin/fengshuiling/julei/16.tif"
CLUSTERS = args.clusters if args.clusters else 5

WIDTH = 128
HEIGHT = 128

with open('colors.json') as clr:
    color_dict = json.load(clr)


def TrainKMeans(img):
    new_width, new_height = calculate_new_size(img)
    image = img.resize((new_width, new_height), Image.ANTIALIAS)
    img_array = np.array(image)
    img_vector = img_array.reshape((img_array.shape[0] * img_array.shape[1], 3))
    '''
    ----------
    Training K-Means Clustering Algorithm
    ----------
    '''
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=0)
    labels = kmeans.fit_predict(img_vector)

    hex_colors = [rgb_to_hex(center) for center in kmeans.cluster_centers_]
    color_name = {}
    for c in kmeans.cluster_centers_:
        h, name = findColorName(c)
        color_name[h] = name

    img_cor = [[*x] for x in img_vector]
    '''
    img_cor is a nested list of all the coordinates (pixel -- RGB value) present in the
    image
    '''
    cluster_map = pd.DataFrame()
    cluster_map['position'] = img_cor
    cluster_map['cluster'] = kmeans.labels_
    cluster_map['x'] = [x[0] for x in cluster_map['position']]
    cluster_map['y'] = [x[1] for x in cluster_map['position']]
    cluster_map['z'] = [x[2] for x in cluster_map['position']]
    cluster_map['color'] = [hex_colors[x] for x in cluster_map['cluster']]
    cluster_map['color_name'] = [color_name[x] for x in cluster_map['color']]
    print(cluster_map)
    return cluster_map, kmeans
##使用了自定义函数calculate_new_size来调整图像的大小。
##将图像的较长尺寸调整为固定尺寸HEIGHT或WIDTH，并重新调整了其他尺寸，同时使高度与图像宽度之比保持恒定。
# 返回TrainKMeans函数，调整图像大小后，我将图像转换为numpy数组，然后将其重塑为3维矢量以表示下一步的RGB值。
def calculate_new_size(image):
    if image.width >= image.height:
        wperc = (WIDTH / float(image.width))
        hsize = int((float(image.height) * float(wperc)))
        new_width, new_height = WIDTH, hsize
    else:
        hperc = (HEIGHT / float(image.height))
        wsize = int((float(image.width) * float(hperc)))
        new_width, new_height = wsize, HEIGHT
    return new_width, new_height

##准备在图像中创建颜色簇。使用KMeans（）函数，我们可以创建群集，其中超参数n_clusters设置为clusters，
# 在程序开始时我们接受的命令行参数，而random_state等于零。接下来，我们将为输入图像文件拟合模型并预测聚类。
# 使用聚类中心（RGB值），我们可以找到聚类代表的相应颜色的十六进制代码，为此使用了rgb_to_hex的自定义函数。
# 它使用matplotlib.colors的to_hex函数。我们已经将RGB值标准化为0到1的范围，然后将它们转换为各自的十六进制代码。
# 现在，我们有了每个颜色簇的十六进制代码。
def rgb_to_hex(rgb):  #Converting our rgb value to hex code.

    hex = color.to_hex([int(rgb[0]) / 255, int(rgb[1]) / 255, int(rgb[2]) / 255])
    print(hex)
    return hex

#findColorName函数中，我们调用了另一个名为get_color_name（）的自定义函数，该函数返回两个值，即aname（实际名称）和cname（最近的颜色名称）。
def findColorName(rgb):
    '''
    Finding color name :: returning hex code and nearest/actual color name
    '''
    aname, cname = get_colour_name((int(rgb[0]), int(rgb[1]), int(rgb[2])))
    hex = color.to_hex([int(rgb[0]) / 255, int(rgb[1]) / 255, int(rgb[2]) / 255])
    if aname is None:
        name = cname
    else:
        name = aname
    return hex, name


def closest_colour(requested_colour):
    min_colors = {}
    for key, name in color_dict['color_names'].items():
        r_c, g_c, b_c = webcolors.hex_to_rgb("#" + key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colors[math.sqrt(rd + gd + bd)] = name
        # print(min(min_colours.keys()))
    return min_colors[min(min_colors.keys())]


def get_colour_name(requested_colour):
    '''
    In this function, we are converting our RGB set to color name using a third
    party module "webcolors".

    RGB set -> Hex Code -> Color Name

    By default, it looks in CSS3 colors list (which is the best). If it cannot find
    hex code in CSS3 colors list, it raises a ValueError which we are handling
    using our own function in which we are finding the closest color to the input
    RGB set.
    '''
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    print(actual_name,closest_name)
    return actual_name, closest_name

# 在此功能中，使用第三方模块webcolors将RGB转换为颜色名称。
# 默认情况下，webcolors函数在CSS3颜色列表中查找。如果无法在其列表中找到颜色，则会引发ValueError，这时使用另一个名为closest_colour（）的自定义函数处理的。
# 在此函数中，我正在计算输入RGB值与JSON中存在的所有RGB值之间的欧式距离。然后，选择并返回距输入RGB值最小距离的颜色。
#
# 在TrainKMeans（）函数中创建的十六进制代码字典及其各自的名称。然后使用img_vector创建了图像中存在的所有RGB点的列表。
# 接下来将初始化一个空的数据框cluster_map，并创建一个名为position的列，该列保存图像和列簇中存在的每个数据点（像素）的RGB值，我存储了每个数据点（像素）被分组到的簇号。
# 然后，在color和color_name列中，我为图像的每个像素存储了十六进制代码及其各自的颜色名称。最后，我们返回了cluster_map数据框和kmeans对象。


##使用散点图绘制了3D空间中图像的每个数据点（像素），并在图像中标识了颜色，并使用饼图显示了图像的颜色分布。
def plotColorClusters(img):
    cluster_map, kmeans = TrainKMeans(img)
    fig = plt.figure()
    ax = Axes3D(fig)
    
    # grouping the data by color hex code and color name to find the total count of
    # pixels (data points) in a particular cluster
    mydf = cluster_map.groupby(['color', 'color_name']).agg({'position':'count'}).reset_index().rename(columns={"position":"count"})
    mydf['Percentage'] = round((mydf['count']/mydf['count'].sum())*100, 1)
    print(mydf)
    
    # Plotting a scatter plot for all the clusters and their respective colors
    ax.scatter(cluster_map['x'], cluster_map['y'], cluster_map['z'], color = cluster_map['color'])
    plt.show()
    
    
    # Subplots with image and a pie chart representing the share of each color identified
    # in the entire photograph/image.
    
    plt.figure(figsize=(14, 8))
    plt.subplot(221)
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(222)
    plt.pie(mydf['count'], labels=mydf['color_name'], colors=mydf['color'], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()


def main():
    img = Image.open(IMG_PATH)
    plotColorClusters(img)

if __name__ == '__main__':
    main()