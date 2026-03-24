import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.ops
import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from source import quality_test
from source.config import locations, names
from source.config.model_register import ModelRegister
from source.data import datatypes

MORE_CERTAIN_POS_FILE = locations.TEMP_DIR / "prec_recall" / "mc.json"
LESS_CERTAIN_NEG_FILE = locations.TEMP_DIR / "prec_recall" / "lc.json"


class GroundTruthJudge:

    def __init__(self, gt_annotations: list[datatypes.Annotation]):
        self.image_ids = set([annotation.image_id for annotation in gt_annotations])
        self.gt_dict = {image_id: [] for image_id in self.image_ids}
        print("Initializing the gt_dict ...")
        for gt in tqdm.tqdm(gt_annotations):
            self.gt_dict[gt.image_id].append(
                torch.tensor([gt.x_min, gt.y_min, gt.x_max, gt.y_max])
            )

    def _split_dt_in_tp_and_fp(
        self, dts: Sequence[datatypes.Annotation], iou_thr: float
    ) -> tuple[list[datatypes.Annotation], list[datatypes.Annotation]]:
        # group detections by image_id (also include images with zero detections)
        dts_dict = {image_id: [] for image_id in self.image_ids}
        for dt in dts:
            dts_dict.setdefault(dt.image_id, []).append(dt)

        marked: list[datatypes.Annotation] = []

        for image_id in self.image_ids:
            current_dts = dts_dict.get(image_id, [])
            if len(current_dts) == 0:
                continue

            # gt boxes for this image (assumes list of 1D tensors [x1,y1,x2,y2])
            gt_list = self.gt_dict.get(image_id, [])
            if len(gt_list) == 0:
                # no GT => everything is FP
                for dt in current_dts:
                    setattr(dt, "is_tp", False)
                    marked.append(dt)
                continue

            # sort dts by score desc (assumes dt.score exists)
            scores = torch.tensor(
                [float(dt.score) for dt in current_dts], dtype=torch.float32
            )
            order = torch.argsort(scores, descending=True).tolist()
            sorted_dts = [current_dts[i] for i in order]

            dt_boxes = torch.tensor(
                [[dt.x_min, dt.y_min, dt.x_max, dt.y_max] for dt in sorted_dts],
                dtype=torch.int32,
            )
            gt_boxes = torch.stack(gt_list, dim=0).to(dtype=torch.float32)

            # IoU matrix: [num_dt, num_gt]
            ious = torchvision.ops.box_iou(dt_boxes, gt_boxes)

            gt_matched = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool)

            for j, dt in enumerate(sorted_dts):
                iou_row = ious[j].clone()

                # prevent matching the same GT twice
                iou_row[gt_matched] = -1.0

                best_iou, best_gt = torch.max(iou_row, dim=0)

                if best_iou.item() >= iou_thr:
                    gt_matched[best_gt] = True
                    setattr(dt, "is_tp", True)
                else:
                    setattr(dt, "is_tp", False)

                marked.append(dt)

        tps = [dt for dt in marked if dt.is_tp]
        fps = [dt for dt in marked if not dt.is_tp]

        return tps, fps

    def create_more_certain_pos(
        self,
        dts: Sequence[datatypes.Annotation],
        score_change: float,
        iou_thr: float = 0.5,
    ) -> list[datatypes.Annotation]:

        tps, fps = self._split_dt_in_tp_and_fp(dts=dts, iou_thr=iou_thr)

        new_scenario = []
        for tp in tps:
            score_to_use = min(1.0, tp.score + score_change)
            tp.score = score_to_use
            new_scenario.append(tp)

        for fp in fps:
            new_scenario.append(fp)

        return new_scenario

    def create_less_certain_neg(
        self,
        dts: Sequence[datatypes.Annotation],
        score_change: float,
        iou_thr: float = 0.5,
    ) -> list[datatypes.Annotation]:

        tps, fps = self._split_dt_in_tp_and_fp(dts=dts, iou_thr=iou_thr)

        new_scenario = []
        for fp in fps:
            score_to_use = max(0.0, fp.score - score_change)
            fp.score = score_to_use
            new_scenario.append(fp)

        for tp in tps:
            new_scenario.append(tp)

        return new_scenario


def get_rec_prec_from_coco(
    detection_json: Path,
    ground_truth_json: Path,
) -> dict[str, np.ndarray]:
    coco_gt = COCO(ground_truth_json)

    with open(detection_json, "r", encoding="utf-8") as f:
        dt_results = json.load(f)
    if not dt_results:
        print("No detection results found.")
        return {
            "recall": np.array([ind * 0.01 for ind in range(0, 101)]),
            "precision": np.array([0.0] * 101),
        }
    coco_dt = coco_gt.loadRes(str(detection_json))

    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt, iouType="bbox")

    # Restrict the evaluation to IoU = 0.5
    coco_eval.params.iouThrs = np.array([0.5])
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    iouThr = 0.50
    areaLbl = "all"
    maxDets = 100
    catId = 1  # = drone

    p = coco_eval.params
    t = np.where(np.isclose(p.iouThrs, iouThr))[0]
    a = [i for i, l in enumerate(p.areaRngLbl) if l == areaLbl][0]
    m = [i for i, d in enumerate(p.maxDets) if d == maxDets][0]
    k = [i for i, cid in enumerate(p.catIds) if cid == catId][0]

    rec = p.recThrs  # (R,)
    prec = coco_eval.eval["precision"][t, :, k, a, m].squeeze()  # (R,)

    save_curve = {
        "recall": np.array(rec.tolist()),
        "precision": np.array(prec.tolist()),
    }

    return save_curve


if __name__ == "__main__":

    ap05 = []
    print("Iterating over the videos ...")
    for dataset_name in tqdm.tqdm(
        ["2019_09_02_GOPR5871_1058_solo", "off_focus_parrot_birds"]
    ):

        plot_file = locations.TEMP_DIR / "prec_recall" / f"{dataset_name}.svg"
        model_state_id = ModelRegister.baseline
        score_change = 1.0

        gt_json = quality_test.get_dvb_ground_truth_coco_json_path(
            video_name=dataset_name
        )

        with open(gt_json, "r", encoding="utf-8") as f:
            gt_results = json.load(f)
        coco_gt = COCO(gt_json)
        gt_annotations = [
            datatypes.Annotation.from_json_dict(gt_dict)
            for gt_dict in gt_results["annotations"]
        ]

        gt_judge = GroundTruthJudge(gt_annotations)

        dt_json = quality_test.get_dvb_detection_coco_json_path(
            dataset_name=dataset_name, model_state_id=model_state_id
        )
        with open(dt_json, "r", encoding="utf-8") as f:
            dt_results = json.load(f)
        if not dt_results:
            print("No detection results found.")
            sys.exit()

        dt_annotations = [
            datatypes.Annotation.from_json_dict(coco_dt) for coco_dt in dt_results
        ]

        more_positive = gt_judge.create_more_certain_pos(
            dt_annotations, score_change=score_change
        )
        more_positive = [detection.to_coco_dict() for detection in more_positive]
        with open(MORE_CERTAIN_POS_FILE, mode="w", encoding="utf-8") as f:
            json.dump(more_positive, f, indent=True)

        less_negative = gt_judge.create_less_certain_neg(
            dt_annotations, score_change=score_change
        )
        less_negative = [detection.to_coco_dict() for detection in less_negative]
        with open(LESS_CERTAIN_NEG_FILE, mode="w", encoding="utf-8") as f:
            json.dump(less_negative, f, indent=True)

        bl_curve = get_rec_prec_from_coco(dt_json, gt_json)
        add_data_curve = get_rec_prec_from_coco(
            quality_test.get_dvb_detection_coco_json_path(
                dataset_name=dataset_name, model_state_id=ModelRegister.additional_data
            ),
            gt_json,
        )
        extra_data_curve = get_rec_prec_from_coco(
            quality_test.get_dvb_detection_coco_json_path(
                dataset_name=dataset_name, model_state_id=ModelRegister.parrot_data
            ),
            gt_json,
        )
        starting_curve = get_rec_prec_from_coco(
            quality_test.get_dvb_detection_coco_json_path(
                dataset_name=dataset_name, model_state_id=ModelRegister.starting_point
            ),
            gt_json,
        )
        mc_curve = get_rec_prec_from_coco(MORE_CERTAIN_POS_FILE, gt_json)
        lc_curve = get_rec_prec_from_coco(LESS_CERTAIN_NEG_FILE, gt_json)

        ap05_bl = bl_curve["precision"].mean()
        ap05_mc = mc_curve["precision"].mean()
        ap05_ad = add_data_curve["precision"].mean()
        ap05_ex = extra_data_curve["precision"].mean()
        ap05_lc = lc_curve["precision"].mean()
        ap05_st = starting_curve["precision"].mean()
        ap05_target = min(ap05_mc, ap05_lc)
        ap05.append({dataset_name: ap05_target})

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            bl_curve["recall"],
            bl_curve["precision"],
            label=f"Baseline AP05 = {ap05_bl:0.2f}",
        )
        ax.plot(
            mc_curve["recall"],
            mc_curve["precision"],
            label=f"+{score_change} Score for True Positives AP05 = {ap05_mc:0.2f}",
        )
        ax.plot(
            lc_curve["recall"],
            lc_curve["precision"],
            label=f"-{score_change} Score for False Positives AP05 = {ap05_lc:0.2f}",
        )
        ax.plot(
            add_data_curve["recall"],
            add_data_curve["precision"],
            label=f"Additional Data AP05 = {ap05_ad:0.2f}",
        )
        ax.plot(
            extra_data_curve["recall"],
            extra_data_curve["precision"],
            label=f"Add. Data + Fixed Wing Drone AP05 = {ap05_ex:0.2f}",
        )
        ax.plot(
            starting_curve["recall"],
            starting_curve["precision"],
            label=f"Starting point model AP05 = {ap05_st:0.2f}",
        )

        plt.title(f"{dataset_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"{dataset_name} completed.")

    with open(
        locations.TEMP_DIR / "prec_recall" / "target.json", mode="w", encoding="utf-8"
    ) as f:
        json.dump(ap05, f, indent=True)
