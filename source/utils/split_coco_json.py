import json

import tqdm

from source.config import locations, names
from source.db import database_manager as dbm

coco_input_file = (
    locations.Results.coco_jsons / "dvb" / "339" / "Drone versus Bird.json"
)

playground = locations.Results.coco_jsons / "dvb" / "339" / "playground"

video_names = names.DroneVsBirdVideos.video_names
image_ids_for_video_index = {
    name: set(
        dbm.get_image_ids_for_video_id(
            video_id=dbm.get_video_id_for_video_name_stem(video_name_stem=name)
        )
    )
    for name in video_names
}

print("Creating the Image id Index ...")
video_index_for_image_id = {}
video_name_index = {}
for i, video_name in tqdm.tqdm(enumerate(video_names)):
    video_name_index[i] = video_name
    for image_id in image_ids_for_video_index[video_name]:
        video_index_for_image_id[image_id] = i

with open(coco_input_file, mode="r", encoding="utf-8") as f:
    raw_data = json.load(f)

print("Splitting the data ...")
detection_data_for_video_index = {}
for row in tqdm.tqdm(raw_data):
    image_id = row["image_id"]
    video_index = video_index_for_image_id[image_id]
    if video_index not in detection_data_for_video_index:
        detection_data_for_video_index[video_index] = [row]
    else:
        detection_data_for_video_index[video_index].append(row)

print("Saving the new json files ...")
for name_index, detections in tqdm.tqdm(detection_data_for_video_index.items()):
    with open(
        playground / f"{video_name_index[name_index]}.json", mode="w", encoding="utf-8"
    ) as of:
        json.dump(detections, fp=of)
