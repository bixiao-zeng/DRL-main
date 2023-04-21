import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np

class plotstyle(object):
    def __init__(self):
        self.font1 = {'family': 'Times New Roman',
                      'weight': 'normal',
                      'size': 85,
                      }  # label
        self.color = ['purple', 'red', 'green', 'blue', 'orange', 'cyan', 'grey']
        self.ticksize = 80
        self.legfont = {'family': 'Times New Roman',
                        'weight': 'normal',
                        'size': 60,
                        }

    def plotscar(self,path,x,y,xlabel,ylabel,legend,title=None):
        plt.figure(num=1, figsize=(27, 30))
        for t in range(len(y)):
            plt.plot(x, y[t], lw=8,label=legend[t])

        plt.xlabel(xlabel, self.font1)
        plt.ylabel(ylabel, self.font1)
        if title != None:
            plt.title(title, self.font1)
        if 'strategy' in path:
            if legend[0] == 'server':
                plt.ylim(3,5)
            if legend[0] == 'client1':
                plt.ylim(0,1)

        plt.legend(prop=self.font1)
        plt.xticks(rotation=0, size=self.ticksize)
        plt.yticks(size=self.ticksize)
        plt.tight_layout()
        plt.savefig(path)
        plt.show()

    def func(self,x, y):

        return np.power(x, 2) + np.power(y, 2)

    def surface(self):

        fig1 = plt.figure()  # 创建一个绘图对象
        ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
        '''X=np.arange(-2,2,0.1)
        Y=np.arange(-2,2,0.1)#创建了从-2到2，步长为0.1的arange对象
        #至此X,Y分别表示了取样点的横纵坐标的可能取值
        #用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点
        X,Y=np.meshgrid(X,Y)'''
        X, Y = np.mgrid[-2:2:40j, -2:2:40j]  # 从-2到2分别生成40个取样坐标，并作满射联合
        Z = self.fun(X, Y)  # 用取样点横纵坐标去求取样点Z坐标
        plt.title("This is main title")  # 总标题
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, alpha=0.5)  # 用取样点(x,y,z)去构建曲面
        ax.set_xlabel('x label', color='r')
        ax.set_ylabel('y label', color='g')
        ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
        plt.show()  # 显示模块中的所有绘图对象
