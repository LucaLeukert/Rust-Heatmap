import asyncio
import io

import matplotlib.pyplot as plt
import numpy as np
import rustplus.api.structures
import seaborn as sns
from PIL import Image
from rustplus import RustSocket
from typing import Dict

raw_map_data: rustplus.api.structures.RustMap
info: rustplus.api.structures.RustInfo
heatmap_data: Dict[str, np.array]


async def main():
    socket = RustSocket("45.88.230.46", "28017", 1, -1, raise_ratelimit_exception=False)
    await socket.connect()
    print(f"Server name: {(await socket.get_info()).name}")

    global raw_map_data, info, heatmap_data
    raw_map_data = await socket.get_raw_map_data()
    info = await socket.get_info()
    heatmap_data = {}

    for member in (await socket.get_team_info()).members:
        heatmap_data[member.name] = np.zeros((raw_map_data.height, raw_map_data.width))

    while True:
        member = await fetch_team_members(socket, raw_map_data, info, heatmap_data)
        await asyncio.sleep(10)


def exit_handler():
    print("Saving image...")
    save_image()


def world_to_map_x(x, rust_map: rustplus.api.structures.RustMap, info: rustplus.api.structures.RustInfo):
    return x * ((rust_map.width - 2 * rust_map.margin) / info.size) + rust_map.margin


def world_to_map_y(y, rust_map: rustplus.api.structures.RustMap, info: rustplus.api.structures.RustInfo):
    n = rust_map.height - 2 * rust_map.margin
    return rust_map.height - (y * (n / info.size) + rust_map.margin)


def add_player_coord(data: np.array,
                     rust_map: rustplus.api.structures.RustMap,
                     info: rustplus.api.structures.RustInfo,
                     x: int, y: int,
                     intensity=1):
    for i in range(-4, 4):
        for j in range(-4, 4):
            map_x = world_to_map_x(x, rust_map, info)
            map_y = world_to_map_y(y, rust_map, info)

            data[int(map_y + i), int(map_x + j)] += intensity


async def fetch_team_members(socket: RustSocket,
                             rust_map: rustplus.api.structures.RustMap,
                             info: rustplus.api.structures.RustInfo,
                             data: Dict[str, np.array]):
    members = (await socket.get_team_info()).members
    for member in members:
        # print(f"Player {member.name} is at {member.x}, {member.y}")
        if member.name in data and data[member.name] is not None:
            add_player_coord(data[member.name], rust_map, info, int(member.x), int(member.y))

    return members


def save_image():
    for name, data in heatmap_data.items():
        if name is None or data is None:
            continue

        map_image = Image.open(io.BytesIO(raw_map_data.jpg_image))
        map_array = np.array(map_image)

        plt.figure(figsize=(map_image.width / 100, map_image.height / 100), dpi=100)
        heatmap = sns.heatmap(data, cbar=False, annot=False, alpha=1, zorder=2)
        heatmap.imshow(map_array, aspect=heatmap.get_aspect(), extent=heatmap.get_xlim() + heatmap.get_ylim(), zorder=1)
        heatmap.set_axis_off()
        plt.savefig('heatmap-' + name + '.png', bbox_inches='tight', pad_inches=0)

        # combined_image = Image.open('heatmap-' + name + '.png')
        # combined_image.show()


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Saving image...")
    save_image()
