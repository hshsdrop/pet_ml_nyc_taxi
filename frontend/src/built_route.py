import folium
import osmnx as ox
import networkx as nx
from typing import Tuple


def find_shortest_route(start_coords: Tuple[float, float],
                        end_coords: Tuple[float, float],
                        place: str = 'Manhattan, New York, United States',
                        mode: str = 'drive',
                        optimizer: str = 'length') -> folium.folium.Map:
    """Function finds the optimal route between two objects by coordinates on the map.
    :param start_coords: Сoordinates of the start of the trip 
    :param end_coords: Сoordinates of the end of the trip 
    :param place: Trip area
    :param mode: Type of trip - bike, drive, walk
    :param optimizer: Optimization goal, time or length
    :return: Shortes route map
    """
    # graph = ox.graph_from_place(place, network_type = mode)
    # orig_node = ox.distance.nearest_nodes(graph, *start_coords)
    # dest_node = ox.distance.nearest_nodes(graph, *end_coords)
    # # orig_node = ox.get_nearest_node(graph, start_coords)
    # # dest_node = ox.get_nearest_node(graph, end_coords)
    # shortest_route = nx.shortest_path(graph, source=orig_node, target=dest_node, weight=optimizer)
    # shortest_route_map = ox.plot_route_folium(graph, shortest_route, tiles='openstreetmap', route_color='#6495ED')
    # start_marker = folium.Marker(location = start_coords, icon = folium.Icon(color='green'), popup='Start')
    # end_marker = folium.Marker(location = end_coords, icon = folium.Icon(color='red'), popup='End')
    # start_marker.add_to(shortest_route_map)
    # end_marker.add_to(shortest_route_map)
    # return shortest_route_map

    print(f'place: {place}')
    print(f'mode: {mode}\n')
    print(f'\nstart_coords_in_br: {start_coords}')
    print(f'end_coords_in_br: {end_coords}\n')
    
    graph = ox.graph_from_place(place, network_type = mode)

    orig_node = ox.distance.nearest_nodes(graph, *start_coords)
    dest_node = ox.distance.nearest_nodes(graph, *end_coords)
    shortest_route = nx.shortest_path(graph, source=orig_node, target=dest_node, weight=optimizer)

    print(f'\norig_node: {orig_node}')
    print(f'dest_node: {dest_node}')
    print(f'shortest_route: {shortest_route}\n')

    gdf_edges = ox.graph_to_gdfs(graph, nodes=False).reset_index()
    gdf_edges.to_file("route_info.json", driver="GeoJSON")
    shortest_route_map = gdf_edges[gdf_edges['u'].isin(shortest_route[:-1]) & gdf_edges['v'].isin(shortest_route[1:])]
    shortest_route_map = shortest_route_map.explore()
    # shortest_route_map = gdf_edges[gdf_edges['u'].isin(shortest_route[:-1]) & gdf_edges['v'].isin(shortest_route[1:])].explore()
    start_marker = folium.Marker(location = start_coords, icon = folium.Icon(color='green'), popup='Start')
    end_marker = folium.Marker(location = end_coords, icon = folium.Icon(color='red'), popup='End')
    start_marker.add_to(shortest_route_map)
    end_marker.add_to(shortest_route_map)
    return shortest_route_map

