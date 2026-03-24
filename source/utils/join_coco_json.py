import json

import tqdm

from source.config import locations, names
from source.config.model_register import ModelRegister
from source.db import database_manager as dbm

if __name__ == "__main__":

    case = str(ModelRegister.parrot_data)

    folder_with_individual_coco_file = (
        locations.Results.coco_jsons / "dvb" / f"{case}" / "box_nms_0.5"
    )
    combined_result_file = folder_with_individual_coco_file / "Drone versus Bird.json"

    video_names = names.DroneVsBirdVideos.video_names

    print("Creating the combined detection result list ...")
    image_data_list = []
    for i, video_name in tqdm.tqdm(enumerate(video_names)):
        with open(
            folder_with_individual_coco_file / f"{video_name}.json",
            mode="r",
            encoding="utf-8",
        ) as f:
            single_video_data = json.load(f)
            image_data_list.extend(single_video_data)

    print("Saving the new json files ...")
    with open(combined_result_file, mode="w", encoding="utf-8") as of:
        json.dump(image_data_list, fp=of)
