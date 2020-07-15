import sys
import numpy as np
import DQN.my_env as my_env

# 传入参数为距离矩阵
def Dijkstra(distinction):
    N = distinction.shape[0]  # 节点个数
    MAX = sys.maxsize  # 初始为最大值

    dists = [MAX for i in range(N)]  # 让左上角的开始点到所有点距离初始为最大值
    nodes = set()  # 存放已经算出最短距离的点,初始集合为空
    parents = [i for i in range(N)]  # 记录路径 初始都为他们本身

    dists[0] = 0  # 到自己距离为0
    min_point = 0  # 距离左上角最近的点,初始为空

    while (len(nodes) < N):  # 最短距离的节点数量没满就继续循环
        nodes.add(min_point)  # 将当前最短距离加入集合
        # 遍历与最短边直接相连的节点
        for i, weight in enumerate(distinction[min_point]):
            if i not in nodes and weight > 0:  # 更新不在最短距离集,且可达的点 的距离
                if (dists[min_point] + distinction[min_point, i] < dists[i]):  # 值得放缩
                    dists[i] = dists[min_point] + distinction[min_point, i]
                    parents[i] = min_point

        # 选出不在最短距离节点集的,但是到根节点距离最小的点作为下一个最小节点
        min_dist = MAX
        for i, weight in enumerate(dists):
            if i not in nodes and weight > 0 and weight < min_dist:
                min_dist = weight
                min_point = i

    print("左上角到右下角的距离为:",dists[N - 1])

    return parents

#得到最短距离的路径
def paths(parents, location):
    paths=[location]
    while parents[location] != 0:
        paths.append(parents[location])
        location=parents[location]
    paths.append(0)
    #反转路径,得到从根路径开始的
    paths=paths[::-1]
    return paths

if __name__ == '__main__':
    env=my_env.Env()
    n = env.n
    distinction = env.distinction[:]
    parents=Dijkstra(distinction)
    N=distinction.shape[0]
    paths=paths(parents,N-1)
    print("路径为:")
    for i in range(len(paths)):
        print("[",paths[i]//n,",",paths[i]%n,"]",",")

