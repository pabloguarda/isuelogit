# Github repos:
# https://github.com/zhouyanupc/subway_beijing/blob/master/assignment_reference.py#L1
# https://github.com/nomoreoneday/Beijing_Subway_route_seach_agent/blob/master/Optimal_route.ipynb
# https://github.com/RyanPeking/Beijing_Subway_Route/blob/master/Beijing_Subway_Route.ipynb

### Set of route to traval between 2 stations in Beijing
### https://map.baidu.com/subway/%E5%8C%97%E4%BA%AC%E5%B8%82/@12960453.129236592,4834588.04436301,18.02z/ccode%3D131%26cname%3D%25E5%258C%2597%25E4%25BA%25AC%25E5%25B8%2582

import requests
import re
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

def request_gis_data(url = ""):
    # 获取URL(高德地图,北京地铁数据):http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json

    path = ""

    if url == "":

        # Locally
        from requests_testadapter import Resp

        class LocalFileAdapter(requests.adapters.HTTPAdapter):
            def build_response_from_file(self, request):
                file_path = request.url[7:]
                with open(file_path, 'rb') as file:
                    buff = bytearray(os.path.getsize(file_path))
                    file.readinto(buff)
                    resp = Resp(buff)
                    r = self.build_response(request, resp)

                    return r

            def send(self, request, stream=False, timeout=None,
                     verify=True, cert=None, proxies=None):

                return self.build_response_from_file(request)

        path = os.getcwd() + "/" + "examples/data/beijing-subway/gis.json"

        requests_session = requests.session()
        requests_session.mount('file://', LocalFileAdapter())
        r = requests_session.get('file://'+ path)

    else:
        # From internet
        url = 'http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json'
        r = requests.get(url)
        # print(r.text) 观察数据结构

    return r

# 遍历数据,组成地点数据结构
def get_lines_stations_info(text):
    # 所有线路信息的dict: key:线路名称,value:站点名称list
    lines_info = {}
    # 所有站点信息的dict: key:站点名称,value:站点坐标(x,y)
    stations_info = {}

    pattern = re.compile('"st".*?"kn"')
    lines_list = pattern.findall(text)
    for i in range(len(lines_list)):
        # 地铁线路名
        pattern = re.compile('"ln":".*?"')
        line_name = pattern.findall(lines_list[i])[0][6:-1] # 获取线路名
        # 站点信息list
        pattern = re.compile('"rs".*?"sp"')
        temp_list = pattern.findall(lines_list[i])
        station_name_list = []
        for j in range(len(temp_list)):
            # 地铁站名
            pattern = re.compile('"n":".*?"')
            station_name = pattern.findall(temp_list[j])[0][5:-1] # 获取站名
            station_name_list.append(station_name)
            # 坐标(x,y)
            pattern = re.compile('"sl":".*?"')
            position = pattern.findall(temp_list[j])[0][6:-1] # 获取坐标str
            position = tuple(map(float,position.split(','))) # 转换为float
            # 将数据加入站点信息dict
            stations_info[station_name] = position
        # 将数据加入地铁线路信息dict
        lines_info[line_name] = station_name_list
    return lines_info, stations_info

def get_neighbor_info(lines_info):
    # 吧str2加入str1站点的邻接表中
    # Let str2 join the adjacency list of the str1 site
    def add_neighbor_dict(info,str1,str2):
        list1 = info.get(str1)
        if not list1:
            list1 = []
        list1.append(str2)
        info[str1] = list1
        return info
    # 根据线路信息,建立站点邻接表dict
    # According to the line information, establish the site adjacency list dict
    neighbor_info = {}
    for line_name,station_list in lines_info.items():
        for i in range(len(station_list)-1):
            sta1 = station_list[i]
            sta2 = station_list[i+1]
            neighbor_info = add_neighbor_dict(neighbor_info,sta1,sta2)
            neighbor_info = add_neighbor_dict(neighbor_info,sta2,sta1)
    return neighbor_info

def plot_network(stations_info, neighbor_info):
    # 画地铁图
    plt.figure(figsize=(20,20)) # 设置宽高
    stations_graph = nx.Graph()
    stations_graph.add_nodes_from(list(stations_info.keys()))
    nx.draw(stations_graph,stations_info,with_labels=True,font_size=5, node_size=2)
    plt.show()
    stations_connection_graph = nx.Graph(neighbor_info)
    nx.draw(stations_connection_graph,stations_info,with_labels=True,font_size=5, node_size=2)
    plt.show()

def plot_network_and_labels(stations_info, neighbor_info):
    # 显示汉字label
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # 画出地铁站点地图
    station_graph = nx.Graph()
    station_graph.add_nodes_from(list(stations_info.keys()))
    plt.figure(figsize=(20, 20))

    # 添加地铁线路
    neighbor_graph = nx.Graph(neighbor_info)
    nx.draw(neighbor_graph, stations_info, with_labels=True, node_size=20, font_size=10, node_color='red')
    plt.show()

# 第一种算法:递归查找所有路径
def get_path_DFS_ALL(lines_info, neighbor_info, from_station, to_station):
    """
    递归算法,本质上是深度优先
    遍历所有路径
    这种情况下,站点间的坐标距离难以转化为可靠的启发函数,所以只用简单的BFS算法
    """
    # 检查输入站点名称
    if not neighbor_info.get(from_station):
        print('起始站点{}不存在,请输入正确的站点名称'.format(from_station))
        return None
    if not neighbor_info.get(to_station):
        print('目的站点{}不存在,请输入正确的站点名称'.format(to_station))
        return None
    path = []
    this_station = from_station
    path.append(this_station)
    neighbors = neighbor_info.get(this_station)
    node = {'pre_station':'',
            'this_station':this_station,
            'neighbors':neighbors,
            'path':path}
    return get_next_station_DFS_ALL(node, neighbor_info, to_station)

def get_next_station_DFS_ALL(node, neighbor_info, to_station):
    neighbors = node.get('neighbors')
    pre_station = node.get('this_station')
    path = node.get('path')
    paths = []
    for i in range(len(neighbors)):
        this_station = neighbors[i]
        if (this_station in path):
            # 如果此站点已经在路径中,说明环路,此路不通
            return None
        if neighbors[i] == to_station:
            # 找到终点,返回路径
            path.append(to_station)
            paths.append(path)
            return paths
        else:
            neighbors_ = neighbor_info.get(this_station).copy()
            neighbors_.remove(pre_station)
            path_ = path.copy()
            path_.append(this_station)
            new_node = {'pre_station': pre_station,
                        'this_station': this_station,
                        'neighbors': neighbors_,
                        'path': path_}
            paths_ = get_next_station_DFS_ALL(new_node, neighbor_info, to_station)
            if paths_:
                paths.extend(paths_)

    return paths

def get_path_BFS(lines_info, neighbor_info, from_station, to_station):
    """
    搜索策略：以站点数量为cost（因为车票价格是按站算的）
    这种情况下，站点间的坐标距离难以转化为可靠的启发函数，所以只用简单的BFS算法
    由于每深一层就是cost加1，所以每层的cost都相同，算和不算没区别，所以省略
    """

    """Search strategy: take the number of stations as cost (because the ticket price is calculated by station)
     In this case, the coordinate distance between the stations is difficult to transform into a reliable heuristic function, so only a simple BFS algorithm
     Since each deep layer is the cost plus 1, the cost of each layer is the same, it is the same as whether it is not counted, so it is omitted"""

    # 检查输入站点名称
    # Check the input site name
    if not neighbor_info.get(from_station):
        # print('起始站点{}不存在,请输入正确的站点名称'.format(from_station))
        print("The starting site {} does not exist, please enter the correct site name".format(from_station))

        return None
    if not neighbor_info.get(to_station):
        # print('目的站点{}不存在,请输入正确的站点名称'.format(to_station))
        print("Destination site {} does not exist, please enter the correct site name")
        return None

    # 搜索节点是个dict，key=站名，value是包含路过的站点list
    nodes = {}
    nodes[from_station] = [from_station]

    while True:
        new_nodes = {}
        for (k, v) in nodes.items():
            neighbor = neighbor_info.get(k).copy()
            if (len(v) >= 2):
                # 不往上一站走
                pre_station = v[-2]
                neighbor.remove(pre_station)
            for station in neighbor:
                # 遍历邻居
                if station in nodes:
                    # 跳过已搜索过的节点
                    continue
                path = v.copy()
                path.append(station)
                new_nodes[station] = path
                if station == to_station:
                    # 找到路径，结束
                    return path
        nodes = new_nodes

    # print('未能找到路径')
    print("Failed to find the path")
    return None

# paths = get_path_BFS(lines_info, neighbor_info, '回龙观', '西二旗')
# print("路径总计{}站。".format(len(paths) - 1))
# print("-".join(paths))

# 第三种算法：以路径路程为cost的启发式搜索

def get_path_Astar(lines_info, neighbor_info, stations_info, from_station, to_station):
    """
    搜索策略：以路径的站点间直线距离累加为cost，以当前站点到目标的直线距离为启发函数
    """
    # 检查输入站点名称
    if not neighbor_info.get(from_station):
        print('起始站点{}不存在,请输入正确的站点名称'.format(from_station))
        return None
    if not neighbor_info.get(to_station):
        print('目的站点{}不存在,请输入正确的站点名称'.format(to_station))
        return None

    # 计算所有节点到目标节点的直线距离，备用
    distances = {}
    x, y = stations_info.get(to_station)
    for (k, v) in stations_info.items():
        x0, y0 = stations_info.get(k)
        l = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        distances[k] = l

    # 已搜索过的节点，dict
    # key=站点名称，value是已知的起点到此站点的最小cost
    searched = {}
    searched[from_station] = 0

    # 数据结构为pandas的dataframe
    # index为站点名称
    # g为已走路径，h为启发函数值（当前到目标的直线距离）
    nodes = pd.DataFrame([[[from_station], 0, 0, distances.get(from_station)]],
                         index=[from_station], columns=['path', 'cost', 'g', 'h'])

    count = 0
    while True:
        if count > 1000:
            break
        nodes.sort_values('cost', inplace=True)
        for index, node in nodes.iterrows():
            count += 1
            # 向邻居中离目的地最短的那个站点搜索
            neighbors = neighbor_info.get(index).copy()
            if len(node['path']) >= 2:
                # 不向这个路径的反向去搜索
                neighbors.remove(node['path'][-2])
            for i in range(len(neighbors)):
                count += 1
                neighbor = neighbors[i]
                g = node['g'] + get_distance(stations_info, index, neighbor)
                h = distances[neighbor]
                cost = g + h
                path = node['path'].copy()
                path.append(neighbor)
                if neighbor == to_station:
                    # 找到目标，结束
                    print('共检索%d次。' % count)
                    return path
                if neighbor in searched:
                    if g >= searched[neighbor]:
                        # 说明现在搜索的路径不是最优，忽略
                        continue
                    else:
                        searched[neighbor] = g
                        # 修改此站点对应的node信息
                        #                         nodes.loc[neighbor, 'path'] = path # 这行总是报错
                        #                         nodes.loc[neighbor, 'cost'] = cost
                        #                         nodes.loc[neighbor, 'g'] = g
                        #                         nodes.loc[neighbor, 'h'] = h
                        # 不知道怎么修改df中的list元素，只能删除再新增行
                        nodes.drop(neighbor, axis=0, inplace=True)
                        row = pd.DataFrame([[path, cost, g, h]],
                                           index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                        nodes = nodes.append(row)

                else:
                    searched[neighbor] = g
                    row = pd.DataFrame([[path, cost, g, h]],
                                       index=[neighbor], columns=['path', 'cost', 'g', 'h'])
                    nodes = nodes.append(row)
            # 这个站点的所有邻居都搜索完了，删除这个节点
            nodes.drop(index, axis=0, inplace=True)

        # 外层for循环只跑第一行数据，然后重新sort后再计算
        continue

    print('未能找到路径')
    return None

def get_distance(stations_info, str1, str2):
    x1, y1 = stations_info.get(str1)
    x2, y2 = stations_info.get(str2)
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def get_distance_metres(lat_1, lng_1, lat_2, lng_2):

    """Source: https://stackoverflow.com/questions/44743075/calculate-the-distance-between-two-coordinates-with-python

    Assume that latitudes and longitudes are in degrees and return distance in metres
    """
    lat_1, lng_1, lat_2, lng_2 = map(math.radians, [lat_1, lng_1, lat_2, lng_2])

    d_lat = lat_2 - lat_1
    d_lng = lng_2 - lng_1

    temp = (
         math.sin(d_lat / 2) ** 2
       + math.cos(lat_1)
       * math.cos(lat_2)
       * math.sin(d_lng / 2) ** 2
    )

    return 1000* 6373.0 * (2 * math.atan2(math.sqrt(temp), math.sqrt(1 - temp)))



def read_beijing_data(folder_path, filenames = ['7.csv','8.csv','9.csv']):

    # https://explorebj.com/subway/

    # filenames = ["7.csv", "8.csv", "9.csv"]
    # filenames = ["7.csv"]
    cols_filename = "fields.csv"

    files_paths = [folder_path + f for f in filenames]
    cols_path = folder_path + cols_filename

    colnames = list(pd.read_csv(cols_path))

    df = pd.concat([pd.read_csv(f, names=colnames) for f in files_paths])

    return df

def get_dictionary_beijing_data():
    from collections import OrderedDict

    codebook = OrderedDict()
    codebook['oline'] = 'Line of origin station'
    codebook['ostation'] = 'Origin station'
    codebook['otime'] = 'Time at origin station'
    codebook['dline'] = 'Line of destination station'
    codebook['dstation'] = 'Destination station'
    codebook['dtime'] = 'Time at destination station'
    codebook['nid'] = '?'
    codebook['ostationid'] = 'Id of origin station'
    codebook['dstationid'] = 'Id of destination station'
    codebook['dtimeid'] = '?'
    codebook['tt'] = 'Travel time [min]'
    codebook['ostationname'] = 'Name of origin station'
    codebook['dstationname'] = 'Name of destination station'
    codebook['otimeid'] = '?'
    codebook['distance'] = 'Distance [m]'
    codebook['speed'] = 'Speed [km/hr]'
    codebook['oindex'] = '?'
    codebook['dindex'] = '?'
    codebook['dtimemin'] = 'Time destination station'
    codebook['dindex1'] = '?'

    for k, v in codebook.items():
        print(k, ':', v)

    return codebook
