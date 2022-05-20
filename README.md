# k-means
使用k-means进行颜色提取，其中color_jsion文件为颜色类别信息，无需更改。
使用k-means2进行运行，使用图像为512×512图像，举例如下：
 ![image](https://user-images.githubusercontent.com/92282483/169440007-5ea2c06a-f369-463b-9ec1-ab449ce85764.png)

结果为：
 ![image](https://user-images.githubusercontent.com/92282483/169439932-ef019390-a823-4dcd-9aee-533202207ab6.png)

聚类颜色的16进制代码为：
     color  color_name  count  Percentage
0  #25292a       Shark  11025        67.3
1  #884545   Ironstone   1562         9.5
2  #a5796e  Coral Tree   1364         8.3
3  #c1b1a9        Silk   1300         7.9
4  #eae8e9  Gray Nurse   1133         6.9
目前分析原因：颗粒周围边界过多，如绿色颗粒无法正确显示。

